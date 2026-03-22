apt-get update && apt-get install -y \
    mesa-utils \
    x11-utils \
    libosmesa6 \
    libosmesa6-dev \
    xvfb
    
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export MESA_GL_VERSION_OVERRIDE=3.3
export QT_X11_NO_MITSHM=1
export QT_QPA_PLATFORM=xcb
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libOSMesa.so.6   # change if different
export DISPLAY=:99

Xvfb :99 -screen 0 1280x1024x24 >/tmp/xvfb-99.log 2>&1 &
export DISPLAY=:99
sleep 0.5

conda activate comp_robotics
pip install imageio imageio-ffmpeg
conda install -c conda-forge opencv ipywidgets matplotlib jupyterlab gymnasium -y

conda install ffmpeg -c conda-forge
pip install lerobot

# inside container as root (or run with sudo)
apt-get update
apt-get install -y --no-install-recommends gnupg2 curl ca-certificates software-properties-common

# add NVIDIA CUDA apt key and repo for Ubuntu 22.04 (adjust if your container is different)
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
  | gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" \
  > /etc/apt/sources.list.d/cuda.list

apt-get update

# install CUDA toolkit (12.6) and a newer compiler
apt-get install -y --no-install-recommends cuda-toolkit-12-6 gcc-11 g++-11 build-essential

# make nvcc visible in PATH for current shell and future shells
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# verify
which nvcc && nvcc --version

# Check https://pytorch.org/get-started/locally/ for your system
pip install "torch>=2.2.1,<2.8.0" "torchvision>=0.21.0,<0.23.0" # --index-url https://download.pytorch.org/whl/cu1XX
pip install ninja "packaging>=24.2,<26.0" # flash attention dependencies
pip install "flash-attn>=2.5.9,<3.0.0" --no-build-isolation
python -c "import flash_attn; print(f'Flash Attention {flash_attn.__version__} imported successfully')"

pip install lerobot[groot]