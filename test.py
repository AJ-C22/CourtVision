import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("torchvision version:", torchvision.__version__)
if torch.cuda.is_available():
    print("CUDA is available. PyTorch was compiled with CUDA support.")
else:
    print("CUDA is not available. PyTorch was not compiled with CUDA support.")