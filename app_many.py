import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import re
import numpy as np
import pickle
from transformers import BertForSequenceClassification, BertTokenizer, RobertaForSequenceClassification, RobertaTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

# 确保下载必要的 NLTK 资源
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

# Streamlit 界面
st.title("AI vs Human Text Classification")
st.write("选择模型并输入文本进行分类。")

# **设备设置**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **加载 BERT 模型**
bert_model_path = "./saved_model"
bert_model = BertForSequenceClassification.from_pretrained(bert_model_path)
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)
bert_model.to(device)
bert_model.eval()

# **加载 RoBERTa 模型**
roberta_model_path = "./roberta_model"
roberta_model = RobertaForSequenceClassification.from_pretrained(roberta_model_path)
roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_model_path)
roberta_model.to(device)
roberta_model.eval()

# **加载 LSTM 模型**
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

lstm_vocab_size = 40157
lstm_embedding_dim = 100
lstm_hidden_dim = 128
lstm_output_dim = 2
lstm_dropout_rate = 0.3

lstm_model = LSTMClassifier(lstm_vocab_size + 1, lstm_embedding_dim, lstm_hidden_dim, lstm_output_dim, lstm_dropout_rate)
lstm_model.load_state_dict(torch.load('./lstm_model.pth', map_location=device))
lstm_model.to(device)
lstm_model.eval()

# 加载 LSTM 词汇表
checkpoint = torch.load('./processed_data.pt', map_location=device)
word_to_idx = checkpoint['word_to_idx']


def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model
# 加载所有模型
logistic_regression_model = load_model('logistic_regression_pipeline.pkl')
naive_bayes_model = load_model('naive_bayes_pipeline.pkl')
xgboost_model = load_model('xgboost_pipeline.pkl')

# **文本预处理**
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

# **BERT / RoBERTa 预处理**
def transformer_preprocess(text, tokenizer, max_length=256):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return encoding['input_ids'].to(device), encoding['attention_mask'].to(device)

# **LSTM 预处理**
def lstm_preprocess(text, word_to_idx, max_len=200):
    tokens = preprocess_text(text)
    numerical = [word_to_idx.get(word, 0) for word in tokens]  # OOV 词替换为 0
    if len(numerical) > max_len:
        numerical = numerical[:max_len]
    else:
        numerical += [0] * (max_len - len(numerical))
    return torch.tensor([numerical], dtype=torch.long).to(device)

#xgboost等模型
def predict_with_model(model, text):
    prediction = model.predict([text])
    probabilities = model.predict_proba([text])
    predicted_class = prediction[0]
    if predicted_class == 1:
        predicted_class = "AI 生成"
    else:
        predicted_class = "人类撰写"
    confidence = np.max(probabilities)  # 置信度是最高概率
    return predicted_class, confidence

def predict(text, model_name):
    if model_name =="逻辑回归":
        logistic_regression_prediction, logistic_regression_confidence = predict_with_model(logistic_regression_model,text)
        return logistic_regression_prediction, logistic_regression_confidence
    elif model_name == "朴素贝叶斯":
        naive_bayes_prediction, naive_bayes_confidence = predict_with_model(naive_bayes_model,text)
        return naive_bayes_prediction,naive_bayes_confidence
    elif model_name == "XGBoost":
        xgboost_prediction, xgboost_confidence = predict_with_model(xgboost_model,text)
        return xgboost_prediction,xgboost_confidence
    elif model_name == "BERT":
        input_ids, attention_mask = transformer_preprocess(text, bert_tokenizer)
        model = bert_model
    elif model_name == "RoBERTa":
        input_ids, attention_mask = transformer_preprocess(text, roberta_tokenizer)
        model = roberta_model
    elif model_name == "LSTM":
        input_tensor = lstm_preprocess(text, word_to_idx)
        with torch.no_grad():
            outputs = lstm_model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            return "AI 生成" if predicted_class == 1 else "人类撰写", confidence

    # BERT / RoBERTa 统一推理
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probabilities = F.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    return "AI 生成" if predicted_class == 1 else "人类撰写", confidence




# **Streamlit 界面交互**
model_choice = st.selectbox("选择模型", ["BERT", "RoBERTa", "LSTM", "逻辑回归", "朴素贝叶斯", "XGBoost"])
user_input = st.text_area("输入文本", "请输入要分类的文本...")

if st.button('分类文本'):
    if user_input:
        result, confidence = predict(user_input, model_choice)
        st.subheader(f"分类结果: {result}")
        st.write(f"置信度: {confidence:.4f}")
    else:
        st.error("请输入文本进行分类！")
