# Use the official CUDA base image from the Docker Hub
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Set DEBIAN_FRONTEND to noninteractive to prevent tzdata configuration prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    gnupg2 \
    zlib1g-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    git \
    liblzma-dev 

# Download Python 3.10 source code
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tar.xz

# Extract Python source code
RUN tar -xf Python-3.10.12.tar.xz && \
    cd Python-3.10.12 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall

# Install pip for Python 3.10
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Install any packages in requirements.txt
COPY requirements.txt .
RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install -r requirements.txt

# Set the working directory to /apps
WORKDIR /apps

# Copy the current directory contents into the container at /apps
COPY ./apps /apps

# Set the command to run when the container starts
CMD ["bash"]
