# Base OS: Ubuntu 22.04 (Jammy)
# If you need CUDA, switch this to:
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------
# 1) System packages + libs
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    make \
    pkg-config \
    wget \
    cmake \
    nano \
    curl \
    git \
    python3 \
    python3-pip \
    xvfb \
    xauth \
    x11-xserver-utils \
    ca-certificates \
    bash \
    # Required libs (your list)
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    mesa-common-dev \
    mesa-utils \
    libosmesa6 \
    libosmesa6-dev \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxkbcommon-x11-0 \
    libdbus-1-3 \
    libglib2.0-0 \
    libfontconfig1 \
    libfreetype6 \
    libx11-xcb1 \
    libxcb-glx0 \
    libxcb-util1 \
    libgl1 \
    libegl1 \
    libglvnd0 \
    libglx0 \
    libglu1-mesa \
    libglew2.2 \
    && rm -rf /var/lib/apt/lists/*

# Convenience: "xvfb-run" is provided by xvfb package on Ubuntu
# Optional but sometimes useful for rendering/GL headless apps:
# RUN apt-get update && apt-get install -y --no-install-recommends libgl1 && rm -rf /var/lib/apt/lists/*

# -----------------------------
# 2) CoppeliaSim installation
# -----------------------------
ENV COPPELIASIM_ROOT=/root/CoppeliaSim
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT


RUN apt-get update && apt-get install -y --no-install-recommends xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz && \
    mkdir -p $COPPELIASIM_ROOT && \
    tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1 && \
    rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

# -----------------------------
# 3) Miniconda install to /opt/conda
# -----------------------------
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:$PATH

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm -f /tmp/miniconda.sh && \
    ${CONDA_DIR}/bin/conda config --set always_yes yes --set changeps1 no && \
    ${CONDA_DIR}/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    ${CONDA_DIR}/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    ${CONDA_DIR}/bin/conda update -q conda && \
    ${CONDA_DIR}/bin/conda clean -a -y

# Use bash login shell so `conda` works in subsequent RUN commands
SHELL ["/bin/bash", "-lc"]

# -----------------------------
# 4) Create comp robotics conda env from yml
# -----------------------------
ARG ENV_YML=environment.yml

WORKDIR /workspace

# Copy only the yml first for caching: env is rebuilt only if yml changes
COPY ${ENV_YML} /workspace/environment.yml

RUN conda env create -f /workspace/environment.yml && conda clean -a -y

# Make the env the default for subsequent RUN steps (name from yml is comp_robotics)
ENV CONDA_DEFAULT_ENV=comp_robotics
ENV PATH=${CONDA_DIR}/envs/comp_robotics/bin:${PATH}

# ----------------------------
# 5) Copy all required code directories
# 6) fdp_mod_rlbench editable install
# 7) lerobot editable install with extras ["smolvla"]
# -----------------------------
# Note: Code directories are mounted at runtime via devcontainer.json
# The pip install -e commands will run via postCreateCommand in devcontainer.json

# -----------------------------
# 8) pip install numpy==1.26.4 (inside conda env)
# -----------------------------

RUN python -m pip install --no-cache-dir numpy==1.26.4

# -----------------------------
# 9) Set useful environment variables
# -----------------------------
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV XDG_RUNTIME_DIR=/tmp/runtime-root
ENV DISPLAY=:99
ENV QT_AUTO_SCREEN_SCALE_FACTOR=1
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV GALLIUM_DRIVER=softpipe
ENV LP_NUM_THREADS=1

# Create XDG runtime directory
RUN mkdir -p /tmp/runtime-root && chmod 700 /tmp/runtime-root

# -----------------------------
# 11) Create startup script to launch Xvfb
# -----------------------------
RUN echo '#!/bin/bash\n\
Xvfb :99 -screen 0 1280x1024x24 -ac +extension GLX +render -noreset &\n\
sleep 2\n\
exec "$@"' > /usr/local/bin/start-xvfb.sh && \
    chmod +x /usr/local/bin/start-xvfb.sh

# -----------------------------
# 10) Auto-activate conda environment on shell startup
# -----------------------------
RUN echo "source ${CONDA_DIR}/bin/activate comp_robotics" >> /root/.bashrc

# Default working directory
WORKDIR /workspace

# (Optional) quick sanity checks at build-time:
RUN python -c "import numpy; print(numpy.__version__)" && \
    python -c "import sys; print(sys.executable)"

ENTRYPOINT ["/usr/local/bin/start-xvfb.sh"]
CMD ["/bin/bash"]

