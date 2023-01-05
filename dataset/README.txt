数据集来自于PTDB2.0
隐式数据中的conn是人工标注的伪连接词，测试时也是不可作为输入的
- explicit.json 显式篇章关系数据
- implicit_train.json 隐式篇章关系的训练集
- implicit_dev.json 隐式篇章关系的验证集
- implicit_test.json 隐式篇章关系的测试集
- new_implicit_train.json 使用了回译的方法对implicit_train进行了数据增强