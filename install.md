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





