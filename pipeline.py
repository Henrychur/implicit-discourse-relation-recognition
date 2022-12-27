import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from Config import config
from Dataset import DiscourseDataset
from Model import DiscourseBert
from utils import test
import warnings
warnings.simplefilter("ignore")  #忽略告警

def train():
    train_dataset = DiscourseDataset(mode="train", max_length=config.max_length, use_explict=False)
    val_dataset = DiscourseDataset(mode="dev", max_length=config.max_length)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    model = DiscourseBert()
    model.to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(config.device)

    test_f1_log = []
    for epoch in range(config.epoch):
        model.train()
        train_loss = 0
        y_true = np.array([])
        y_pred = np.array([])
        for batch in train_dataloader:
            optimizer.zero_grad()
            if config.backbone == "bert-base-uncased" or config.backbone == "bert-large-uncased" \
                or config.backbone == "microsoft/deberta-v3-base" or config.backbone == "microsoft/deberta-v3-large":
                labels = batch[3].to(config.device)
                arg = (batch[0].to(config.device), batch[1].to(config.device), batch[2].to(config.device))
            elif config.backbone == "roberta-base" or config.backbone == "roberta-large":
                labels = batch[2].to(config.device)
                arg = (batch[0].to(config.device), batch[1].to(config.device))
            outputs = model(arg)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            y_true = np.append(y_true, labels.cpu().numpy())
            y_pred = np.append(y_pred, torch.argmax(outputs, dim=-1).cpu().numpy())
        print("epoch: {}, train loss: {}, train acc: {}, train f1: {}".format(epoch, train_loss/len(train_dataloader), accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")))


        model.eval()
        val_loss = 0
        y_true = np.array([])
        y_pred = np.array([])
        with torch.no_grad():
            for batch in val_dataloader:
                if config.backbone == "bert-base-uncased" or config.backbone == "bert-large-uncased" \
                    or config.backbone == "microsoft/deberta-v3-base" or config.backbone == "microsoft/deberta-v3-large":
                    labels = batch[3].to(config.device)
                    arg = (batch[0].to(config.device), batch[1].to(config.device), batch[2].to(config.device))
                elif config.backbone == "roberta-base" or config.backbone == "roberta-large" :
                    labels = batch[2].to(config.device)
                    arg = (batch[0].to(config.device), batch[1].to(config.device))
                outputs = model(arg)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                y_true = np.append(y_true, labels.cpu().numpy())
                y_pred = np.append(y_pred, torch.argmax(outputs, dim=-1).cpu().numpy())
        print("epoch: {}, val loss: {}, val acc: {}, val f1: {}".format(epoch, val_loss/len(val_dataloader), accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")))                
        test_f1_log.append(test(model))
    # torch.save(model.state_dict(), config.backbone + "_model.pt")

if __name__ == "__main__":
    train()