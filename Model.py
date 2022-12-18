import torch
import torch.nn as nn
from transformers import AutoModel
from Config import config

class DiscourseBert(nn.Module):
    def __init__(self):
        super(DiscourseBert, self).__init__()
        self.backbone = AutoModel.from_pretrained(config.backbone)
        if config.backbone == "bert-base-uncased" or config.backbone == "roberta-base":
            output_dim = 768
        elif config.backbone == "bert-large-uncased" or config.backbone == "roberta-large":
            output_dim = 1024
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, 1024),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(1024, 4),
        )

    def forward(self, arg):
        if config.backbone == "bert-base-uncased" or config.backbone == "bert-large-uncased":
            res = self.backbone(input_ids=arg[0], token_type_ids=arg[1], attention_mask=arg[2])
        elif config.backbone == "roberta-base" or config.backbone == "roberta-large":
            res = self.backbone(input_ids=arg[0], attention_mask=arg[1])
        out = torch.mean(res.last_hidden_state, dim=1)
        out = self.classifier(out)
        return out