import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pickle

# 读取数据
data = pd.read_csv('processed_data.csv')

# 删除 NaN 并确保文本是字符串类型
data = data.dropna(subset=['cleaned'])
data['cleaned'] = data['cleaned'].astype(str)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data['cleaned'], data['label'], test_size=0.2, random_state=42)

# **创建 Pipeline**

# 逻辑回归分类器
lr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),  # Tfidf 向量化
    ('lr', LogisticRegression(max_iter=1000))  # 逻辑回归
])

# 朴素贝叶斯分类器
nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),  # Tfidf 向量化
    ('nb', MultinomialNB())  # 朴素贝叶斯
])

# XGBoost 分类器
xgb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),  # Tfidf 向量化
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))  # XGBoost
])

# **训练与评估**
# 训练并评估逻辑回归
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)
print("逻辑回归分类结果:")
print(classification_report(y_test, y_pred_lr))

# 训练并评估朴素贝叶斯
nb_pipeline.fit(X_train, y_train)
y_pred_nb = nb_pipeline.predict(X_test)
print("朴素贝叶斯分类结果:")
print(classification_report(y_test, y_pred_nb))

# 训练并评估 XGBoost
xgb_pipeline.fit(X_train, y_train)
y_pred_xgb = xgb_pipeline.predict(X_test)
print("XGBoost 分类结果:")
print(classification_report(y_test, y_pred_xgb))

# **保存模型**
with open('logistic_regression_pipeline.pkl', 'wb') as f:
    pickle.dump(lr_pipeline, f)

with open('naive_bayes_pipeline.pkl', 'wb') as f:
    pickle.dump(nb_pipeline, f)

with open('xgboost_pipeline.pkl', 'wb') as f:
    pickle.dump(xgb_pipeline, f)

print("模型训练完成，并已保存")
