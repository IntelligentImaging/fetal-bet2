import argparse
import logging
import os
import sys
import time

import torch
from torch.optim import SGD
from monai.losses import DiceCELoss

from utils import read_config
from model_zoo import get_network
from data_generator_imagine import FetalDataLoader

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


def train(args):
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    config = read_config(args.cfg, mode="train")

    if not os.path.exists(config.saved_model_path):
        os.makedirs(config.saved_model_path)

    logging.basicConfig(
        filename=os.path.join(config.saved_model_path, "log_train.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(str(config))

    # =========================
    # Load data (train only)
    # =========================
    fetal_data = FetalDataLoader(config, Train=True)
    train_dataloader = fetal_data.load_data()

    # =========================
    # Load model
    # =========================
    model = get_network(config)
    device = args.device

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    if args.pretrained is not None:
        load_pretrained_weights(model, args.pretrained, device)
        logging.info(f"Loaded pretrained weights from {args.pretrained}")

    # =========================
    # Optimizer
    # =========================
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

    # =========================
    # Loss
    # =========================
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

    max_epochs = config.max_epochs
    save_interval = 2
    log_interval = args.log_interval   # print loss every N iterations

    # =========================
    # Training loop
    # =========================
    logging.info("-" * 30 + " training starts " + "-" * 30)
    step_start = time.time()

    loss_window = []

    for epoch in range(max_epochs):
        logging.info(f"Epoch {epoch + 1}/{max_epochs}")
        model.train()

        epoch_loss = 0.0
        step = 0
        num_steps = len(train_dataloader)

        for batch_data in train_dataloader:
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val

            # ===== record loss window =====
            loss_window.append(loss_val)

            # ===== iteration-level logging (window average) =====
            if step % log_interval == 0:
                avg_loss = sum(loss_window) / len(loss_window)
                logging.info(
                    f"Epoch {epoch + 1}/{max_epochs} "
                    f"| Iter {step}/{num_steps} "
                    f"| avg loss (last {len(loss_window)} iters) = {avg_loss:.6f}"
                )
                loss_window.clear()  # clear window

        epoch_loss /= step
        logging.info(f"Epoch {epoch + 1} average loss: {epoch_loss:.6f}")

        # =========================
        # Save model every save_interval epochs
        # =========================
        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(
                config.saved_model_path,
                f"checkpoint_epoch_{epoch + 1:03d}.pth"
            )
            torch.save(model.state_dict(), save_path)
            logging.info(f"Saved model: {save_path}")

    # =========================
    # Save last model
    # =========================
    save_last_path = os.path.join(
        config.saved_model_path,
        "checkpoint_last.pth"
    )
    torch.save(model.state_dict(), save_last_path)
    logging.info(f"Saved last model: {save_last_path}")

    train_time = time.time() - step_start
    logging.info(f"Training completed in {train_time:.2f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--cfg',
        type=str,
        required=True,
        help='path to config file, e.g. config_imagine.yml (3D) or config_imagine_2D.yml (2D); '
             'the model dimensionality is decided entirely by this file, not by this script'
    )

    parser.add_argument(
        '--n_gpu',
        type=int,
        default=2,
        help='total gpu number'
    )

    parser.add_argument(
        '--pretrained',
        type=str,
        default=None,
        help='path to a pretrained checkpoint to fine-tune from (e.g. Docker/src/models/AttUNet.pth '
             'for config_imagine_2D.yml, or AttUNet3D.pth for config_imagine.yml); '
             'leave unset to train from scratch'
    )

    parser.add_argument(
        '--cuda_visible_devices',
        type=str,
        default=None,
        help='value for the CUDA_VISIBLE_DEVICES env var, e.g. "0,1,2"; leave unset to use the current environment'
    )

    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='print loss every N iterations'
    )

    parser.add_argument(
        '--deterministic',
        type=int,
        default=1,
        help='whether use deterministic training'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=1234,
        help='random seed'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='device to use'
    )

    args = parser.parse_args()
    train(args)
