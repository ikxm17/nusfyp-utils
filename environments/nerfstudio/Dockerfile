FROM ghcr.io/nerfstudio-project/nerfstudio:latest

# apt update and install all packages
RUN apt-get update && \
    apt-get install -y \
    imagemagick \
    vim \
    tmux \
    wget \
    curl \
    unzip \
    git \
    gh \
    tldr \
    man && \
    # clean up apt cache
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace