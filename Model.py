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
            nn.Linear(output_dim*2, 1024),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(1024, 4),
        )

    def forward(self, arg1, arg2):
        if config.backbone == "bert-base-uncased" or config.backbone == "bert-large-uncased":
            res1 = self.backbone(input_ids=arg1[0], token_type_ids=arg1[1], attention_mask=arg1[2])
            res2 = self.backbone(input_ids=arg2[0], token_type_ids=arg2[1], attention_mask=arg2[2])
        elif config.backbone == "roberta-base" or config.backbone == "roberta-large":
            res1 = self.backbone(input_ids=arg1[0], attention_mask=arg1[1])
            res2 = self.backbone(input_ids=arg2[0], attention_mask=arg2[1])
        output1 = torch.mean(res1.last_hidden_state, dim=1)
        output2 = torch.mean(res2.last_hidden_state, dim=1)
        out = torch.cat((output1, output2), dim = -1)
        out = self.classifier(out)
        return out