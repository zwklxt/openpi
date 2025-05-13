## Set train and inference environment

follow the task description
#### 1. clone the repo
```bash
#clone the repo and update the submodule..
git clone https://github.com/zwklxt/openpi.git
git submodule update --init --recursive

```

#### 2. install the training environment package
install the training environment package use uv 
```bash
#1. install uv if your environment not installed.
curl -LsSf https://astral.sh/uv/install.sh | sh 
#or
wget -qO- https://astral.sh/uv/install.sh | sh

#2. install traing environments dependence package.
GIT_LFS_SKIP_SMUDGE=1 uv sync 
# some package cannot install auto,you should install it manually.
source .venv/bin/activate
uv pip install av=14.0.1
```

#### 3. install the inference environment package
install the inference environment package use uv
```bash
# All cmd run in openpi path:
# Create virtual environment
uv venv --python 3.11 examples/piper_real/.venv
source examples/piper_real/.venv/bin/activate
# generate requirement.txt
uv pip compile examples/piper_real/requirements.in -o examples/piper_real/requirements.txt --python-version 3.11
UV_HTTP_TIMEOUT=200 uv pip sync examples/piper_real/requirements.txt
UV_HTTP_TIMEOUT=200 uv pip install -e packages/openpi-client

```




