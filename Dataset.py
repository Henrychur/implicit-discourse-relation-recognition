import torch
import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from Config import config
import numpy as np

class DiscourseDataset(Dataset):
    '''
        隐式篇章关系的数据集
    '''
    def __init__(self,mode="train", max_length=30, muti_label=False, use_explict=False):
        assert mode in ["train","dev","test"], "mode must be train, dev or test"
        self.mode = mode
        self.explicit_data_path = None
        if config.modelingMethod == "prompt_93":
            self.conns = self.getConnLabel()
        if use_explict:
            self.explicit_data_path = "./dataset/explicit.json"
        self.data_path = f"./dataset/implicit_{mode}.json"
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(config.backbone)
        self.data = self.load_data(muti_label=muti_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if config.backbone == "bert-base-uncased" or config.backbone == "bert-large-uncased"\
            or config.backbone == "microsoft/deberta-v3-base" or config.backbone == "microsoft/deberta-v3-large":
            if config.modelingMethod == "prompt_93":
                arg1, arg2, label, conn = self.data[index]
            else:    
                arg1, arg2, label = self.data[index]
            arg_input_ids = torch.cat((arg1["input_ids"][0], arg2["input_ids"][0]), dim=0)
            arg_attention_mask = torch.cat((arg1["attention_mask"][0], arg2["attention_mask"][0]), dim=0)
            arg_token_type_ids = torch.cat((arg1["token_type_ids"][0], arg2["token_type_ids"][0]), dim=0)
            if config.modelingMethod == "prompt_93":
                return arg_input_ids, arg_token_type_ids, arg_attention_mask, conn, label
            else:
                return arg_input_ids, arg_token_type_ids, arg_attention_mask, label

        elif config.backbone == "roberta-base" or config.backbone == "roberta-large":
            if config.modelingMethod == "prompt_93":
                arg1, arg2, label, conn = self.data[index]
            else:
                arg1, arg2, label = self.data[index]
            arg_input_ids = torch.cat((arg1["input_ids"][0], arg2["input_ids"][0]), dim=0)
            arg_attention_mask = torch.cat((arg1["attention_mask"][0], arg2["attention_mask"][0]), dim=0)
            if config.modelingMethod == "prompt_93":
                return arg_input_ids, arg_attention_mask, conn, label
            else:
                return arg_input_ids, arg_attention_mask, label


    def load_data(self, muti_label=False):
        data = []
        with open(self.data_path,"r") as f:
            jsonFile = json.load(f)
            for item in jsonFile:
                arg1 = self.tokenizer(item["arg1"], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
                arg2 = self.tokenizer(item["arg2"], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
                if muti_label:
                    label = [int(i) for i in item["label"]]
                else:
                    label = int(item["label"][0])
                if config.modelingMethod == "prompt_93":
                    conn = self.conns.index(item["conn"])
                    data.append((arg1, arg2, label, conn))
                else:
                    data.append((arg1, arg2, label))
        if self.explicit_data_path is not None:
            with open(self.explicit_data_path,"r") as f:
                jsonFile = json.load(f)
                for item in jsonFile:
                    arg1 = self.tokenizer(item["arg1"], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
                    arg2 = self.tokenizer(item["arg2"], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
                    label = int(item["label"][0])
                    if config.modelingMethod == "prompt_93":
                        conn = self.conns.index(item["conn"])
                        data.append((arg1, arg2, label, conn))
                    else:
                        data.append((arg1, arg2, label))
        return data

    def getConnLabel(self):
        # -------------------------- #
        # 加载数据集，获取映射关系
        # -------------------------- # 
        conn_label_mapping = {}
        file_list = ["implicit_train", "implicit_dev", "implicit_test", "explicit"] if config.use_explict else ["implicit_train", "implicit_dev", "implicit_test"]
        for filename in file_list:
            with open(f"dataset/{filename}.json", "r", encoding="utf-8") as f:
                jsonf = json.load(f)
                for item in jsonf:
                    if item["conn"] not in conn_label_mapping.keys():
                        conn_label_mapping[item["conn"]] = np.array([0, 0, 0, 0])
                    for id in item["label"]:
                        conn_label_mapping[item["conn"]][id] += 1
        
        return list(conn_label_mapping.keys())

def dataAnalysis():
    import matplotlib.pyplot as plt
    mode = "test"
    tokenizer = AutoTokenizer.from_pretrained(config.backbone)
    length_ls = []
    with open(f"./dataset/implicit_{mode}.json","r") as f:
        jsonFile = json.load(f)
        for item in jsonFile:
            arg1 = tokenizer(item["arg1"])
            arg2 = tokenizer(item["arg2"])
            length_ls.append(len(arg1["input_ids"]))
            length_ls.append(len(arg2["input_ids"]))
    plt.hist(length_ls, bins=100)
    plt.savefig(f"./dataset/{mode}_length.png")
    print("max length:", max(length_ls))
    print("min length:", min(length_ls))
    print("mean length:", sum(length_ls)/len(length_ls))

def json2csv():
    import pandas as pd
    lines = []
    with open("dataset/implicit_train.json","r") as f:
        jsonFile = json.load(f)
        for item in jsonFile:
            arg1 = item["arg1"]
            arg2 = item["arg2"]
            label = item["label"][0]
            lines.append(arg1+"\n")
            lines.append(arg2+"\n")
    with open("implicit_train.txt", "w") as f:
        for line in lines:
            f.write(line)

if __name__ == "__main__":
    json2csv()
    