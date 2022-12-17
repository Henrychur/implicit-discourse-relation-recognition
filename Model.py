import torch
import torch.nn as nn
from transformers import AutoModel
from Config import config

class DiscourseBert(nn.Module):
    def __init__(self):
        super(DiscourseBert, self).__init__()
        self.bert = AutoModel.from_pretrained(f'bert-{config.scale}-uncased')
        if config.scale == "base":
            output_dim = 768
        elif config.scale == "large":
            output_dim = 1024
        self.classifier = nn.Sequential(
            nn.Linear(output_dim*2, 1024),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(1024, 4),
        )

    def forward(self, arg1, arg2):
        res1 = self.bert(input_ids=arg1[0], token_type_ids=arg1[1], attention_mask=arg1[2])
        res2 = self.bert(input_ids=arg2[0], token_type_ids=arg2[1], attention_mask=arg2[2])
        out = torch.cat((res1.pooler_output, res2.pooler_output), dim = -1)
        out = self.classifier(out)
        return out