import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from Config import config
from Dataset import DiscourseDataset
import json
import numpy as np


def test(model):
    '''
        利用测试集测试模型地macroF1
        需要注意的是数据集中有的样本有多个标签，只要预测其中一个就算正确
    '''
    test_dataset = DiscourseDataset(mode="test", muti_label=True, max_length=config.max_length)
    model.eval()
    y_true = np.array([])
    y_pred = np.array([])
    with torch.no_grad():
        for data in test_dataset:
            if config.backbone == "bert-base-uncased" or config.backbone == "bert-large-uncased" \
                or config.backbone == "microsoft/deberta-v3-base"  or config.backbone == "microsoft/deberta-v3-large":
                labels = data[3]
                arg = (data[0].unsqueeze(dim=0).to(config.device), data[1].unsqueeze(dim=0).to(config.device), data[2].unsqueeze(dim=0).to(config.device))
            elif config.backbone == "roberta-base" or config.backbone == "roberta-large":
                labels = data[2]
                arg = (data[0].unsqueeze(dim=0).to(config.device), data[1].unsqueeze(dim=0).to(config.device))
            outputs = model(arg)
            pred = torch.argmax(outputs, dim=-1).cpu().numpy()
            y_pred = np.append(y_pred, pred)
            if pred[0] in labels:
                y_true = np.append(y_true, pred)
            else:
                y_true = np.append(y_true, labels[0])
    print("test_accuracy_score: ",accuracy_score(y_true, y_pred), "test_f1_score: ",f1_score(y_true, y_pred, average="macro"))
    return f1_score(y_true, y_pred, average="macro")

def getConnLabelMapping():
    '''
        获取连接词和标签的映射关系
        考虑到一个连接词会对应多种标签，选择频率最高的
        return:
            conn_label_mapping: dict, key: conn, value: label
    '''
    # -------------------------- #
    # 1. 加载数据集，获取映射关系
    # -------------------------- # 
    conn_label_mapping = {}
    for filename in ["implicit_train", "explicit"]:
        with open(f"dataset/{filename}.json", "r", encoding="utf-8") as f:
            jsonf = json.load(f)
            for item in jsonf:
                if item["conn"] not in conn_label_mapping.keys():
                    conn_label_mapping[item["conn"]] = np.array([0, 0, 0, 0])
                for id in item["label"]:
                    conn_label_mapping[item["conn"]][id] += 1
    # -------------------------- #
    # 2. 将标签进行归一化
    # -------------------------- #
    for conn in conn_label_mapping.keys():
        conn_label_mapping[conn] = conn_label_mapping[conn] / np.sum(conn_label_mapping[conn])
    
    return conn_label_mapping

if __name__ == "__main__":
    getConnLabelMapping()
