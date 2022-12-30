import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    train_acc, train_f1 = [], []
    val_acc, val_f1 = [], []
    test_acc, test_f1 = [], []
    log_name = "roberta_large_v2.out"
    with open(log_name) as fp:
        for line in fp:
            if "train loss" in line:
                train_acc.append(float(line.split(" ")[-4][:-1]))
                train_f1.append(float(line.split(" ")[-1]))
            elif "val loss" in line:
                val_acc.append(float(line.split(" ")[-4][:-1]))
                val_f1.append(float(line.split(" ")[-1]))
            elif "test_accuracy_score" in line:
                test_acc.append(float(line.split()[-3]))
                test_f1.append(float(line.split()[-1]))
    print("f1 score: ", sorted(test_f1))
    print(sum(test_f1)/len(test_f1))
    # plt.plot(train_acc[:20], label="train_acc")
    # plt.plot(val_acc[:20], label="val_acc")
    # plt.plot(val_f1[:20], label="val_f1")
    # plt.plot()
    # plt.legend()
    # plt.savefig(log_name[:-4]+".png")