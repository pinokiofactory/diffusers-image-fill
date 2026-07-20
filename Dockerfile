# Base image with CUDA 12.4
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Update and Install packages
RUN apt-get update -y && apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6

# Define environment variables for UID and GID
ENV PUID=${PUID:-1000}
ENV PGID=${PGID:-1000}

# Create a group with the specified GID
RUN groupadd -g "${PGID}" appuser
# Create a user with the specified UID and GID
RUN useradd -m -s /bin/sh -u "${PUID}" -g "${PGID}" appuser

WORKDIR /app

# Install main application dependencies
COPY ./requirements.cuda.txt ./requirements.cuda.txt
RUN pip install --no-cache-dir -r ./requirements.cuda.txt

# Install Torch, Torchvision, and Torchaudio for CUDA 12.4
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

RUN chown -R appuser:appuser /app

# delete redundant requirements.cuda.txt
RUN rm ./requirements.cuda.txt

#Run application as non-root
USER appuser

# Copy diffusers image fill application code
COPY . ./diffusers-image-fill

EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"

WORKDIR /app/diffusers-image-fill

# Run diffusers-image-fill Python application
CMD ["python3", "./app.py"]