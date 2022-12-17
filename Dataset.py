import torch
import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from Config import config

class DiscourseDataset(Dataset):
    '''
        隐式篇章关系的数据集
    '''
    def __init__(self,mode="train", max_length=30):
        assert mode in ["train","dev","test"], "mode must be train, dev or test"
        self.data_path = f"./dataset/implicit_{mode}.json"
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(f'bert-{config.scale}-uncased')
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        arg1, arg2, label = self.data[index]
        arg1_input_ids = arg1["input_ids"][0]
        arg1_token_type_ids = arg1["token_type_ids"][0]
        arg1_attention_mask = arg1["attention_mask"][0]
        arg2_input_ids = arg2["input_ids"][0]
        arg2_token_type_ids = arg2["token_type_ids"][0]
        arg2_attention_mask = arg2["attention_mask"][0]
        return arg1_input_ids, arg1_token_type_ids, arg1_attention_mask, arg2_input_ids, arg2_token_type_ids, arg2_attention_mask, label

    def load_data(self):
        data = []
        with open(self.data_path,"r") as f:
            jsonFile = json.load(f)
            for item in jsonFile:
                arg1 = self.tokenizer(item["arg1"], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
                arg2 = self.tokenizer(item["arg2"], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
                label = int(item["label"][0])
                if len(item["label"])>1:
                    print("muti")
                    print(item["label"])
                data.append((arg1, arg2, label))
        return data

if __name__ == "__main__":
    dataset = DiscourseDataset(mode="train")
