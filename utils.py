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
    if config.modelingMethod == "prompt_93":
        connLabelMapping = getConnLabelMapping()
        conn_list = list(connLabelMapping.keys())
    with torch.no_grad():
        for data in test_dataset:
            if config.backbone == "bert-base-uncased" or config.backbone == "bert-large-uncased" \
                or config.backbone == "microsoft/deberta-v3-base"  or config.backbone == "microsoft/deberta-v3-large":
                if config.modelingMethod == "prompt_93":
                    labels = data[4]
                else:
                    labels = data[3]
                arg = (data[0].unsqueeze(dim=0).to(config.device), data[1].unsqueeze(dim=0).to(config.device), data[2].unsqueeze(dim=0).to(config.device))
            elif config.backbone == "roberta-base" or config.backbone == "roberta-large":
                if config.modelingMethod == "prompt_93":
                    labels = data[3]
                else:
                    labels = data[2]
                arg = (data[0].unsqueeze(dim=0).to(config.device), data[1].unsqueeze(dim=0).to(config.device))
            outputs = model(arg)
            pred = torch.argmax(outputs, dim=-1).cpu().numpy()
            if config.modelingMethod == "prompt_93":
                pred = np.array([np.argmax(connLabelMapping[conn_list[pred[0]]])])
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
    file_list = ["implicit_train", "implicit_dev", "implicit_test", "explicit"] if config.use_explict else ["implicit_train", "implicit_dev", "implicit_test", ]
    for filename in file_list:
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

def translate():
    from transformers import pipeline
    from tqdm import tqdm
    en_fr_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr", device=0)
    fr_en_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en", device=0)
    newJsonFile = []
    with open("./dataset/implicit_train.json","r") as f:
        jsonFile = json.load(f)
        for item in tqdm(jsonFile):
            arg1 = item["arg1"]
            arg2 = item["arg2"]
            label = item["label"]
            conn = item["conn"]
            # translate
            arg1 = en_fr_translator(arg1)[0]["translation_text"]
            arg1 = fr_en_translator(arg1)[0]["translation_text"]

            arg2 = en_fr_translator(arg2)[0]["translation_text"]
            arg2 = fr_en_translator(arg2)[0]["translation_text"]

            new_item = {"arg1":arg1, "arg2":arg2, "label":label, "conn":conn}
            newJsonFile.append(new_item)
    with open("./dataset/new_implicit_train.json", "w") as f:
        json_data = json.dumps(newJsonFile)
        f.write(json_data)


if __name__ == "__main__":
    translate()
