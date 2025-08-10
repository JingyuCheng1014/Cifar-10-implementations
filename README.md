## 0. Clone this Project:

```bash
git clone https://github.com/JingyuCheng1014/Cifar-10-implementations.git
cd /path/to/Cifar-10-implementations
```

## 1. Set up and enter a environment:

Create a new environment (**conda**):

```bush
conda create -n vit-cifar10 python=3.10 -y
conda activate vit-cifar10
python -m pip install -U pip setuptools wheel
```

## 2. Install dependencies: 

Download cuda12.1 at: https://developer.nvidia.com/cuda-12-1-0-download-archive

```bash
# Using Linux System with Navida GPU: 
pip install -r requirements-cu121.txt

# or using CPU:
pip install -r requirements-cpu.txt
```

A quick way to check if the GPU/CUDA is recognized:

```bash
python - <<'PY'
import torch; print("torch:", torch.__version__, "cuda available?", torch.cuda.is_available())
PY
```

## 3. Training

 (will automatically download CIFAR-10 to ./data/)

```bash
python train.py
```

## 4. Training with Ablation Study
adjust the parameters (patch size, dim, depth, heads) and compare training results.
```bash
python train_ablation.py
```

## 5. Evaluation/Verification
Load trained data from checkpoints and varify accuracy.
```bash
python eval.py --ckpt checkpoints/tinycnn_cifar10_best.pth

# Or self custom parameters
python eval.py --ckpt checkpoints/tinycnn_cifar10_best.pth --data_root ./data --batch_size 256 --num_workers 8

```

