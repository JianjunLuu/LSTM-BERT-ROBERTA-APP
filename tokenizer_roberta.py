import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from data_processing import data
from transformers import RobertaTokenizer

vocab_file = ' roberta-base/vocab.json'

merges_file = ' roberta-base/merges.txt'


# 数据划分
X_train, X_test, y_train, y_test = train_test_split(
    data['cleaned'].tolist(),
    data['label'].tolist(),
    test_size=0.3,
    random_state=42
)

# 初始化 BERT Tokenizer
tokenizer = RobertaTokenizer(vocab_file, merges_file)
max_length = 256
batch_size = 16

# 自定义 Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',  # 填充到 max_length
            truncation=True,       # 截断到 max_length
            return_attention_mask=True,#Attention Mask，用于指示填充部分（0）和有效部分（1）。
            return_tensors='pt'    # 返回 PyTorch 张量
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # 去掉 batch 维度
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 批次合并函数
def collate_fn(batch):
    # 用 pad_sequence 处理 input_ids 和 attention_mask，确保批次内序列对齐
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


# 构建数据集和 DataLoader
train_dataset = TextDataset(X_train, y_train, tokenizer, max_length)
test_dataset = TextDataset(X_test, y_test, tokenizer, max_length)

# 只保留一个 DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

