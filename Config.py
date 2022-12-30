import torch
class config:
    max_length = 64
    batch_size = 128
    device = torch.device("cuda")
    epoch = 20
    lr = 1e-5 # 2e-5 5e-6 2e-6
    backbone = "roberta-base" # bert-base-uncased, bert-large-uncased, roberta-base, roberta-large microsoft/deberta-v3-base
    modelingMethod = "prompt" # classification, prompt, interaction, prompt_93
    use_explict = True