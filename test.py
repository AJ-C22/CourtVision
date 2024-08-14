import torch
print(torch.cuda.is_available())  # Check if CUDA is available
print(torch.rand(5, 3))  # A simple tensor operation to see if torch runs