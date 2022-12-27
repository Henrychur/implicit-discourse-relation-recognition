import torch
import torch.nn as nn
from transformers import AutoModel
from Config import config
from utils import getConnLabelMapping

class DiscourseBert(nn.Module):
    def __init__(self):
        super(DiscourseBert, self).__init__()
        self.backbone = AutoModel.from_pretrained(config.backbone)
        # ------------------------ # 
        # 选择骨干网络,决定输出维度
        # ------------------------ # 
        if config.backbone == "bert-base-uncased" or config.backbone == "roberta-base" or config.backbone == "microsoft/deberta-v3-base":
            output_dim = 768
        elif config.backbone == "bert-large-uncased" or config.backbone == "roberta-large" or config.backbone == "microsoft/deberta-v3-large":
            output_dim = 1024
        
        # ------------------------ # 
        # 根据建模方式，确定输出维度
        # ------------------------ # 
        if config.modelingMethod == "classification":
            self.classifier = nn.Sequential(
                nn.Linear(output_dim, 1024),
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(1024, 4),
            )
        if config.modelingMethod == "prompt":
            self.connLabelMapping = getConnLabelMapping()
            self.classifier = nn.Sequential(
                nn.Linear(output_dim, 1024),
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(1024, len(self.connLabelMapping)),
            )

    def forward(self, arg):
        # ------------------------ #
        # 输入骨干网络进行特征提取
        # ------------------------ #
        if config.backbone == "bert-base-uncased" or config.backbone == "bert-large-uncased" \
            or config.backbone == "microsoft/deberta-v3-base" or config.backbone == "microsoft/deberta-v3-large":
            res = self.backbone(input_ids=arg[0], token_type_ids=arg[1], attention_mask=arg[2])
        elif config.backbone == "roberta-base" or config.backbone == "roberta-large":
            res = self.backbone(input_ids=arg[0], attention_mask=arg[1])

        # ------------------------ #
        # 根据建模方式输入分类器
        # ------------------------ #
        if config.modelingMethod == "classification":
            out = torch.mean(res.last_hidden_state, dim=1)
            out = self.classifier(out)
        elif config.modelingMethod == "prompt":
            # 首先预测连接词，取出连接词位置的特征向量
            tmp_out = res.last_hidden_state[:, config.max_length, :]
            tmp_out = self.classifier(tmp_out)
            # 根据概率分布将conn的概率转换为label的概率
            out = torch.zeros((tmp_out.size()[0], 4)).to(config.device)
            for i, key in enumerate(self.connLabelMapping.keys()):
                for j, value in enumerate(self.connLabelMapping[key]):
                    out[:, j] += tmp_out[:, i] * value
        return out