# ─────────────────────────────────────────────────────
#  NEXUS TRAFFIC AI — Dockerfile
#  Base: Ubuntu 22.04 with Python 3.11 + SUMO + PyTorch
# ─────────────────────────────────────────────────────
FROM ubuntu:22.04

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV SUMO_HOME=/usr/share/sumo
ENV PYTHONPATH=/usr/share/sumo/tools:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# ── System packages ───────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-dev \
    sumo \
    sumo-tools \
    sumo-doc \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python python3.11 1

# ── Python packages ───────────────────────────────────
RUN pip3 install --upgrade pip
RUN pip3 install \
    torch --index-url https://download.pytorch.org/whl/cpu \
    numpy \
    matplotlib \
    pandas

# ── App setup ─────────────────────────────────────────
WORKDIR /app

# Copy all project files
COPY . .

# Create required output folders
RUN mkdir -p models plots maps maps_images

# ── Expose port for website ───────────────────────────
EXPOSE 8000

# ── Default: show help menu ───────────────────────────
CMD ["python3", "docker_run.py"]
