import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from string import punctuation
from sklearn.pipeline import Pipeline
import pickle

# 预处理函数
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
    return text


# 加载模型（假设你之前保存了 pipeline 模型）
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# 示例文本
sample_text = "This is a sample text for classification."

# 加载所有模型
logistic_regression_model = load_model('logistic_regression_pipeline.pkl')
naive_bayes_model = load_model('naive_bayes_pipeline.pkl')
xgboost_model = load_model('xgboost_pipeline.pkl')

# 预处理文本
preprocessed_text = preprocess_text(sample_text)

# 推理函数
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

# 逻辑回归预测
logistic_regression_prediction, logistic_regression_confidence = predict_with_model(logistic_regression_model, preprocessed_text)
print(f"逻辑回归 - 预测结果: {logistic_regression_prediction}, 置信度: {logistic_regression_confidence:.4f}")

# 朴素贝叶斯预测
naive_bayes_prediction, naive_bayes_confidence = predict_with_model(naive_bayes_model, preprocessed_text)
print(f"朴素贝叶斯 - 预测结果: {naive_bayes_prediction}, 置信度: {naive_bayes_confidence:.4f}")

# XGBoost预测
xgboost_prediction, xgboost_confidence = predict_with_model(xgboost_model, preprocessed_text)
print(f"XGBoost - 预测结果: {xgboost_prediction}, 置信度: {xgboost_confidence:.4f}")