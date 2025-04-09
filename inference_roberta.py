import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

#模型保存看jupyter notebook

# 设备设置
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 加载 RoBERTa 模型和 tokenizer
model_path = "./roberta_model"
model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained(model_path)

# 将模型移动到设备并设置为评估模式
model.to(device)
model.eval()

# 预处理文本函数
def preprocess_text(text, tokenizer, max_length=256):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"  # 返回 PyTorch 张量
    )
    return encoding['input_ids'].to(device), encoding['attention_mask'].to(device)

# 推理函数
def predict(text):
    input_ids, attention_mask = preprocess_text(text, tokenizer)

    with torch.no_grad():  # 关闭梯度计算
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()  # 取最大概率的类别

    return "AI 生成" if prediction == 1 else "人类撰写"

# 测试推理
if __name__ == "__main__":
    test_texts = [
        "The advancements in AI have led to remarkable breakthroughs in various fields.",
        "This article discusses the impact of social media on human behavior."
    ]

    for text in test_texts:
        result = predict(text)
        print(f"输入文本: {text}\n预测结果: {result}\n")
