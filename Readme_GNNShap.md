# CUDA Extension Setup

The GNNShap uses a **custom CUDA C++ extension** that is compiled at runtime using PyTorch.  
Users must explicitly provide CUDA configuration information before running the project.

---

## Requirements

- NVIDIA GPU with CUDA support  
- CUDA Toolkit installed  
- PyTorch with CUDA enabled  

---


## Environment Setup
```bash
conda create -n heteroshap python=3.10 -y
conda activate heteroshap
# Install CUDA
conda install -c nvidia cuda-toolkit -y
# Install PyTorch (CUDA 12.9)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
# Install remaining dependencies
pip install scikit-learn
pip install tqdm
pip install pandas
pip install ninja

```

---



## Required Configuration

You must provide the following values:

| Parameter | Description |
|-----------|-------------|
| `cuda_home` | Path to your CUDA installation |
| `torch_cuda_arch_list` | GPU compute capability |

---

## Find `cuda_home`

### Windows
```cmd
where nvcc
```

Example output:
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe
```

Use:
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
```

### Linux
```bash
which nvcc
```

Example output:
```
/usr/local/cuda/bin/nvcc
```

Use:
```
/usr/local/cuda
```

---

## Find `torch_cuda_arch_list`

Run:
```bash
nvidia-smi
```

```cmd
where nvcc
```

Pass both "cuda_home" and "torch_cuda_arch_list" values in explainer config files.

---

## GNNShap Configuration

GNNShap supports both CUDA and CPU-based samplers, allowing users to switch between them:
```json
"sampler_type": "cuda_based"  // or "cpu"
```

**Note:** It is recommended to select `"cuda_based"` for speed and reproducibility with the original code logic.

### Number of Samples

Users also need to specify the number of samples in each coalition:
```json
"nsamples": 15000
```

If the number of players is really high, increase it to 25000 or 30000.

---
