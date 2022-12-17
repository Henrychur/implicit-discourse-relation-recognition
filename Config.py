import torch
class config:
    max_length = 25
    batch_size = 32
    device = torch.device("cuda")
    epoch = 100
    lr = 2e-5 # 2e-5 5e-6 2e-6
    scale = "base" # base or large