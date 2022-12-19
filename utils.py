import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from Config import config
from Model import DiscourseBert
from Dataset import DiscourseDataset


def test(model):
    '''
        利用测试集测试模型地macroF1
        需要注意的是数据集中有的样本有多个标签，只要预测其中一个就算正确
    '''
    test_dataset = DiscourseDataset(mode="test", muti_label=True)
    model.eval()
    y_true = np.array([])
    y_pred = np.array([])
    with torch.no_grad():
        for data in test_dataset:
            if config.backbone == "bert-base-uncased" or config.backbone == "bert-large-uncased":
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

if __name__ == "__main__":
    model = DiscourseBert()
    model.to(config.device)
    model.load_state_dict(torch.load("roberta-large_model.pt"))
    test(model)
