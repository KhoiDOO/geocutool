# geocutool

# Setup

```bash
pip install -e . --no-build-isolation

# or 

pip install git+https://github.com/KhoiDOO/geocutool.git --no-build-isolation
```

## Tested Env

```bash
conda create -c conda-forge -n geocutool python=3.10 gxx_linux-64=13 gcc_linux-64=13 -y
conda activate geocutool

conda install nvidia::cuda-toolkit==12.8.2 -y

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

```
