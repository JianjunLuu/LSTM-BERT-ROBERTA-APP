import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from tqdm import tqdm

# ========== 设置设备 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 加载模型和 tokenizer ==========
model_path = "C:/Users/ASUS/BERT/saved_model"  # 替换为你保存模型的路径
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# ========== 定义 Dataset ==========
class CSVDatasetForBERT(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=256):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["text"]
        label = int(self.data.iloc[idx]["label"])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label)
        }

# ========== 加载测试数据 ==========
csv_test_path = "C:/Users/ASUS/BERT/attack/ai_rewritten_in_human_style.csv"
test_dataset = CSVDatasetForBERT(csv_test_path, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ========== 模型评估 ==========
predictions, true_labels = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        preds = torch.argmax(logits, dim=1)

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# ========== 输出结果 ==========
accuracy = accuracy_score(true_labels, predictions)
report = classification_report(true_labels, predictions, target_names=["Human-written", "AI-written"])

print(f"\nTest Accuracy: {accuracy:.4f}")
print("Classification Report:\n")
print(report)
