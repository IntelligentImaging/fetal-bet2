"""
Train a 3D-UNet brain-extraction model on the SVR-reconstructed volumes
(train_data_svr.csv), with periodic full-volume testing on test_data_svr.csv
and TensorBoard logging of the training/testing curves.

Reference structure: ../train.py
"""

import argparse
import logging
import os
import sys
import time

import torch
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter

from monai.data import decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType
from monai.utils import set_determinism

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import read_config
from model_zoo import get_network
from data_generator_svr import load_train_data, load_test_data

torch.backends.cudnn.benchmark = True


def load_pretrained_weights(model, pretrained_path, device):
    state_dict = torch.load(pretrained_path, map_location=device)
    is_wrapped = isinstance(model, torch.nn.DataParallel)
    has_module_prefix = next(iter(state_dict)).startswith("module.")

    if is_wrapped and not has_module_prefix:
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}
    elif not is_wrapped and has_module_prefix:
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)


def run_test(model, test_dataloader, inferer, loss_function, post_pred, post_label,
             dice_metric, device):
    model.eval()
    loss_sum = 0.0
    num_cases = 0
    first_batch = None

    with torch.no_grad():
        for test_data in test_dataloader:
            inputs = test_data["image"].to(device)
            labels = test_data["label"].to(device)

            outputs = inferer(inputs, model)
            loss_sum += loss_function(outputs, labels).item()
            num_cases += 1

            outputs_list = [post_pred(o) for o in decollate_batch(outputs)]
            labels_list = [post_label(l) for l in decollate_batch(labels)]
            dice_metric(y_pred=outputs_list, y=labels_list)

            if first_batch is None:
                first_batch = (inputs.cpu(), outputs_list[0].cpu(), labels_list[0].cpu())

    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    mean_loss = loss_sum / max(num_cases, 1)
    model.train()
    return mean_loss, mean_dice, first_batch


def log_example_slices(writer, first_batch, epoch):
    """Log the mid-axial slice of image/prediction/label for one test case."""
    if first_batch is None:
        return
    image, pred, label = first_batch
    mid = image.shape[-1] // 2

    img_slice = image[0, 0, :, :, mid]
    pred_slice = pred[1, :, :, mid]  # foreground channel after to_onehot
    label_slice = label[1, :, :, mid]

    def normalize(x):
        x = x - x.min()
        denom = x.max()
        return x / denom if denom > 0 else x

    writer.add_image("test/image", normalize(img_slice).unsqueeze(0), epoch)
    writer.add_image("test/prediction", pred_slice.unsqueeze(0), epoch)
    writer.add_image("test/label", label_slice.unsqueeze(0), epoch)


def train(args):
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    if args.deterministic:
        set_determinism(seed=args.seed)

    config = read_config(args.cfg, mode="train")

    if not os.path.exists(config.saved_model_path):
        os.makedirs(config.saved_model_path)

    log_dir = os.path.join(config.saved_model_path, "tb_logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    logging.basicConfig(
        filename=os.path.join(config.saved_model_path, "log_train.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(str(config))
    logging.info(f"TensorBoard logs: {log_dir}")

    # Load data
    train_dataloader = load_train_data(config)
    test_dataloader = load_test_data(config)

    # Load model
    model = get_network(config)
    device = args.device

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    if args.pretrained is not None:
        load_pretrained_weights(model, args.pretrained, device)
        logging.info(f"Loaded pretrained weights from {args.pretrained}")

    # Optimizer
    if config.optimizer == "SGD":
        optimizer = SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=0.0001,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4
        )

    # Loss / metric / inferer
    loss_function = DiceCELoss(
        include_background=config.include_background,
        to_onehot_y=True,
        softmax=True,
        squared_pred=True,
        batch=False,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
        lambda_dice=0.6,
        lambda_ce=0.4,
    )

    dice_metric = DiceMetric(include_background=config.include_background, reduction="mean")
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=config.out_channels)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=config.out_channels)])

    inferer = SlidingWindowInferer(
        roi_size=tuple(config.img_size),
        sw_batch_size=config.sw_batch_size,
        overlap=config.sw_overlap,
        mode="gaussian",
    )

    max_epochs = config.max_epochs
    val_interval = config.val_interval
    save_interval = config.save_interval
    log_interval = args.log_interval

    # Training loop
    logging.info("-" * 30 + " training starts " + "-" * 30)
    step_start = time.time()

    loss_window = []
    global_step = 0
    best_dice = -1.0

    for epoch in range(max_epochs):
        logging.info(f"Epoch {epoch + 1}/{max_epochs}")
        model.train()

        epoch_loss = 0.0
        step = 0
        num_steps = len(train_dataloader)

        for batch_data in train_dataloader:
            step += 1
            global_step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val

            writer.add_scalar("train/iter_loss", loss_val, global_step)

            loss_window.append(loss_val)

            if step % log_interval == 0:
                avg_loss = sum(loss_window) / len(loss_window)
                logging.info(
                    f"Epoch {epoch + 1}/{max_epochs} "
                    f"| Iter {step}/{num_steps} "
                    f"| avg loss (last {len(loss_window)} iters) = {avg_loss:.6f}"
                )
                loss_window.clear()

        epoch_loss /= step
        writer.add_scalar("train/epoch_loss", epoch_loss, epoch + 1)
        logging.info(f"Epoch {epoch + 1} average loss: {epoch_loss:.6f}")

        # Periodic testing
        if (epoch + 1) % val_interval == 0 or (epoch + 1) == max_epochs:
            test_loss, test_dice, first_batch = run_test(
                model, test_dataloader, inferer, loss_function,
                post_pred, post_label, dice_metric, device,
            )
            writer.add_scalar("test/loss", test_loss, epoch + 1)
            writer.add_scalar("test/dice", test_dice, epoch + 1)
            log_example_slices(writer, first_batch, epoch + 1)
            logging.info(
                f"[Test] Epoch {epoch + 1}: loss = {test_loss:.6f}, dice = {test_dice:.6f}"
            )

            if test_dice > best_dice:
                best_dice = test_dice
                best_path = os.path.join(config.saved_model_path, "checkpoint_best.pth")
                torch.save(model.state_dict(), best_path)
                logging.info(f"New best dice {best_dice:.6f}, saved: {best_path}")

        # Periodic checkpointing
        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(
                config.saved_model_path,
                f"checkpoint_epoch_{epoch + 1:03d}.pth"
            )
            torch.save(model.state_dict(), save_path)
            logging.info(f"Saved model: {save_path}")

    # Save last model
    save_last_path = os.path.join(config.saved_model_path, "checkpoint_last.pth")
    torch.save(model.state_dict(), save_last_path)
    logging.info(f"Saved last model: {save_last_path}")

    train_time = time.time() - step_start
    logging.info(f"Training completed in {train_time:.2f} seconds")
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--cfg',
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_inference_3d.yml"),
        help='path to config file (default: config_inference_3d.yml next to this script, '
             'also used by inference_3d.py)'
    )

    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu number')

    parser.add_argument(
        '--pretrained',
        type=str,
        default=None,
        help='path to a pretrained checkpoint to fine-tune from; leave unset to train from scratch'
    )

    parser.add_argument(
        '--cuda_visible_devices',
        type=str,
        default=None,
        help='value for the CUDA_VISIBLE_DEVICES env var, e.g. "0,1"; leave unset to use the current environment'
    )

    parser.add_argument('--log_interval', type=int, default=10, help='print loss every N iterations')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', help='device to use')

    args = parser.parse_args()
    train(args)
