import pandas as pd
import re
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

# 函数：检查文本长度，过滤掉短文本
def check_length(text):
    if len(text) < 10:  
        return None  
    else:
        return text 

# 函数：评分编码，将大于等于4的评分设为好评（1），否则为差评（0）
def ratings_encoder(rating):
    if rating >= 4:
        return 1
    else:
        return 0

# 函数：清洗文本（去标点、特殊字符，分词，去停用词）
def clean_text(text):
    # 删除标点符号和特殊字符
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
    # 使用jieba库进行分词
    words = jieba.cut(text)
    # 停用词列表
    stop_words = set(['的', '是', '在', '有', '和', '了', '不', '人', '这', '也', '要', '去'])
    # 删除停用词
    r_words = [word for word in words if word not in stop_words]
    return ' '.join(r_words)

# 读取数据集
pd_ratings = pd.read_csv('/share/home/nieyiran/nieyiran/Introduction to Data Science/datasets/ratings.csv')

# 删除重复行
pd_ratings = pd_ratings.drop_duplicates()

# 去除“comment”为空的行
pd_ratings = pd_ratings.dropna(subset=['comment'])

# 填补缺失评分值：为rating、rating_env、rating_flavor、rating_service填入平均值
for col in ['rating', 'rating_env', 'rating_flavor', 'rating_service']:
    average_value = pd_ratings[col].mean()
    pd_ratings[col] = pd_ratings[col].fillna(round(average_value, 1))

# 去除较短的文本
pd_ratings['comment'] = pd_ratings['comment'].apply(check_length)
pd_ratings = pd_ratings.dropna(subset=['comment'])

# 评分编码：将大于等于4的评分定义为好评(1)，否则为差评(0)
pd_ratings['rating'] = pd_ratings['rating'].apply(ratings_encoder)

# 清洗评论文本
pd_ratings['comment'] = pd_ratings['comment'].apply(clean_text)

# 保存完整的原始数据，用于后续做可视化
pd_ratings.to_csv("original_data.csv", index=False)

# 划分训练集和测试集
train_data, test_data = train_test_split(pd_ratings, test_size=0.2, random_state=42)

# 使用 Count Vectorizer 向量化评论文本
count_vectorizer = CountVectorizer(max_features=500)
train_count_matrix = count_vectorizer.fit_transform(train_data['comment'])
test_count_matrix = count_vectorizer.transform(test_data['comment'])

# 将 Count Vectorizer 特征矩阵转换为 DataFrame
train_count_df = pd.DataFrame(train_count_matrix.toarray(), columns=count_vectorizer.get_feature_names_out())
test_count_df = pd.DataFrame(test_count_matrix.toarray(), columns=count_vectorizer.get_feature_names_out())

# 保存 Count Vectorizer 特征文件
train_count_with_rating = pd.concat([train_data['rating'].reset_index(drop=True), train_count_df], axis=1)
test_count_with_rating = pd.concat([test_data['rating'].reset_index(drop=True), test_count_df], axis=1)
print(train_count_with_rating.head())
train_count_with_rating.to_csv("train_data_with_count_features.csv", index=False)
test_count_with_rating.to_csv("test_data_with_count_features.csv", index=False)

# 使用 TF-IDF Vectorizer 向量化评论文本
tfidf_vectorizer = TfidfVectorizer(max_features=500)
train_tfidf_matrix = tfidf_vectorizer.fit_transform(train_data['comment'])
test_tfidf_matrix = tfidf_vectorizer.transform(test_data['comment'])

# 将 TF-IDF 特征矩阵转换为 DataFrame
train_tfidf_df = pd.DataFrame(train_tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
test_tfidf_df = pd.DataFrame(test_tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# 保存 TF-IDF 特征文件
train_tfidf_with_rating = pd.concat([train_data['rating'].reset_index(drop=True), train_tfidf_df], axis=1)
test_tfidf_with_rating = pd.concat([test_data['rating'].reset_index(drop=True), test_tfidf_df], axis=1)
train_tfidf_with_rating.to_csv("train_data_with_tfidf_features.csv", index=False)
test_tfidf_with_rating.to_csv("test_data_with_tfidf_features.csv", index=False)

print("数据已成功保存为：original_data.csv, train_data_with_count_features.csv, test_data_with_count_features.csv, train_data_with_tfidf_features.csv, test_data_with_tfidf_features.csv。")