import torch
class config:
    max_length = 128
    batch_size = 32
    device = torch.device("cuda")
    epoch = 20
    lr = 2e-5 # 2e-5 5e-6 2e-6
    backbone = "roberta-base" # bert-base-uncased, bert-large-uncased, roberta-base, roberta-large