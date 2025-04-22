import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import re
from transformers import BertForSequenceClassification, BertTokenizer
from openai import OpenAI

# -------------------- NLTK 初始化 --------------------
def ensure_resources_downloaded():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
ensure_resources_downloaded()

# -------------------- Streamlit UI --------------------
st.title("🧠 AI vs Human Text Classification")
st.write("请选择模型，输入文本，判断是否为 AI 生成，并使用 DeepSeek 提供依据和高亮。")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- 加载 BERT 模型 --------------------
bert_model_path = "./saved_model"
bert_model = BertForSequenceClassification.from_pretrained(bert_model_path)
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)
bert_model.to(device)
bert_model.eval()

# -------------------- 加载 LSTM 模型 --------------------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout_rate):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        x = self.fc(self.dropout(lstm_out))
        return x

lstm_model = LSTMClassifier(40158, 100, 128, 2, 0.3)
lstm_model.load_state_dict(torch.load('./lstm_model.pth', map_location=device))
lstm_model.to(device)
lstm_model.eval()

checkpoint = torch.load('./processed_data.pt', map_location=device)
word_to_idx = checkpoint['word_to_idx']

# -------------------- 文本预处理 --------------------
def remove_punc(text):
    return ''.join([char for char in text if char not in punctuation])

def remove_stop(text):
    stops = set(stopwords.words('english'))
    return " ".join([word for word in text.split() if word.lower() not in stops])

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = remove_punc(text)
    text = remove_stop(text)
    return word_tokenize(text)

def bert_preprocess(text, tokenizer, max_length=256):
    encoding = tokenizer.encode_plus(
        text, add_special_tokens=True, max_length=max_length,
        padding="max_length", truncation=True, return_tensors="pt"
    )
    return encoding['input_ids'].to(device), encoding['attention_mask'].to(device)

def lstm_preprocess(text, word_to_idx, max_len=200):
    tokens = preprocess_text(text)
    numerical = [word_to_idx.get(word, 0) for word in tokens]
    if len(numerical) > max_len:
        numerical = numerical[:max_len]
    else:
        numerical += [0] * (max_len - len(numerical))
    return torch.tensor([numerical], dtype=torch.long).to(device)

# -------------------- 推理函数 --------------------
def predict(text, model_name):
    if model_name == "BERT":
        input_ids, attention_mask = bert_preprocess(text, bert_tokenizer)
        with torch.no_grad():
            outputs = bert_model(input_ids, attention_mask=attention_mask)
            probabilities = F.softmax(outputs.logits, dim=1)
    elif model_name == "LSTM":
        input_tensor = lstm_preprocess(text, word_to_idx)
        with torch.no_grad():
            outputs = lstm_model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)

    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0, predicted_class].item()
    return "AI 生成" if predicted_class == 1 else "人类撰写", confidence

# -------------------- DeepSeek 接口 --------------------
client = OpenAI(
    api_key="sk-02f85c18644c4cabbcad55f0961122f0",
    base_url="https://api.deepseek.com"
)

def deepseek_detect(text):
    prompt = (
        "请判断下面的文本中哪些部分可能是由AI生成的，并用[AI]和[/AI]标记包裹，同时给出理由：\n\n"
        f"{text}\n\n"
        "请返回格式：[高亮文本]\n解释：xxxx"
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"调用失败：{e}"

def highlight_ai_text(deepseek_response):
    explanation = ""
    if "解释：" in deepseek_response:
        parts = deepseek_response.split("解释：")
        text_part = parts[0].strip()
        explanation = "解释：" + parts[1].strip()
    else:
        text_part = deepseek_response

    highlighted = text_part.replace("[AI]", "<span style='background-color:#ffdddd'>") \
                           .replace("[/AI]", "</span>")
    return highlighted, explanation

# -------------------- Streamlit 主交互 --------------------
model_choice = st.selectbox("选择模型", ["BERT", "LSTM"])
user_input = st.text_area("输入文本", "请输入要分类的文本...")

if st.button("分类文本"):
    if user_input.strip():
        result, confidence = predict(user_input, model_choice)
        st.subheader(f"分类结果: {result}")
        st.write(f"置信度: {confidence:.4f}")
    else:
        st.error("请输入文本进行分类！")

if st.button("使用 DeepSeek 检测并解释"):
    if user_input.strip():
        with st.spinner("DeepSeek 检测中..."):
            deepseek_result = deepseek_detect(user_input)
            highlighted_text, explanation = highlight_ai_text(deepseek_result)

        st.markdown("#### 🕵️ DeepSeek 检测结果：")
        st.markdown(highlighted_text, unsafe_allow_html=True)
        st.markdown(f"**{explanation}**")
    else:
        st.error("请输入文本以使用 DeepSeek 检测！")
