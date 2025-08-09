import torch
print(torch.__version__)  # Should show 2.7.0+cu126
print(torch.cuda.is_available())  # Should show True
print(torch.version.cuda)  # Should show 12.6
print(torch.cuda.get_device_name(0))  # Should show NVIDIA GeForce RTX 2070 SUPER