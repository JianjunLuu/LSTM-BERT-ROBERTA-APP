import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from data_processing import data
from tokenizer_roberta import train_loader,test_loader


# 初始化 RoBERTa 模型
model = RobertaForSequenceClassification.from_pretrained(' roberta-base', num_labels=2)

# 配置优化器和设备
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 训练模型
model.train()
EPOCHS = 3

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    epoch_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 前向传播
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        # outputs 是一个元组，我们需要从中获取损失和 logits
        loss = outputs[0]  # 获取第一个元素，即损失
        logits = outputs[1]  # 获取第二个元素，即 logits

        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        epoch_loss += loss.item()  # 累加损失

    print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(train_loader)}")

# 测试模型
model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs[0]
        preds = torch.argmax(logits, dim=1)

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# 计算精度和其他评估指标
print("Classification Report:\n", classification_report(true_labels, predictions))
print("Accuracy: ", accuracy_score(true_labels, predictions))

# 保存模型
model.save_pretrained("./roberta_model")
#准确率0.983