import pandas as pd
import re
from nltk.corpus import stopwords
from string import punctuation
from collections import Counter
import torch
from sklearn.model_selection import train_test_split


# 数据读取
data = pd.read_csv('./AI_Human.csv')

# 数据采样与清洗
ai_samples = data[data['generated'] == 1]
human_samples = data[data['generated'] == 0]
data = pd.concat([ai_samples.sample(n=5000, random_state=42), human_samples.sample(n=5000, random_state=42)])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# 清洗函数
def remove_punc(text):
    return ''.join([char for char in text if char not in punctuation])

def remove_stop(text):
    stops = set(stopwords.words('english'))
    return " ".join([word for word in text.split() if word.lower() not in stops])

# 文本清洗
data['cleaned'] = data['text'].str.lower()
data['cleaned'] = data['cleaned'].apply(lambda x: re.sub(r'https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE))
data['cleaned'] = data['cleaned'].apply(lambda x: re.sub(r'<.*?>', '', x))
data['cleaned'] = data['cleaned'].apply(remove_punc)
data['cleaned'] = data['cleaned'].apply(remove_stop)

data = data[['cleaned', 'generated']]
data.rename(columns={'generated': 'label'}, inplace=True)

# 保存清洗后的数据
data.to_csv('processed_data.csv', index=False)
print("数据处理完成，已保存为 processed_data.csv")