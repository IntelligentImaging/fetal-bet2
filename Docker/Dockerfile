# Use a base image with PyTorch and MONAI pre-installed
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Install dependencies from requirements.txt
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt -v

# Copy your project files into the container
COPY src /app/src

# COPY preserves the source files' host permissions, which may not be world-readable;
# make sure the app is usable regardless of what user `docker run --user` picks.
RUN chmod -R a+rX /app/src

# Specify the command to run when the container starts
ENTRYPOINT ["python", "/app/src/codes/inference.py"]
