import pandas as pd
import re
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

def check_length(text):
    if len(text) < 10:  
        return None  
    else:
        return text 

def ratings_encoder(rating):
    if rating > 3.5:
        return 1
    else:
        return 0

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

pd_ratings = pd.read_csv('./datasets/ratings.csv')

# 删除重复行
pd_ratings = pd_ratings.drop_duplicates()

# 去除“comment”为空的行
pd_ratings =  pd_ratings.dropna(subset=['comment'])

# 计算rating属性的平均值
average_rating = pd_ratings['rating'].mean()
rounded_average_rating = round(average_rating, 1)
pd_ratings['rating'] = pd_ratings['rating'].fillna(rounded_average_rating)

# 计算rating_env属性的平均值
average_rating_env = pd_ratings['rating_env'].mean()
rounded_average_rating_env = round(average_rating_env, 1)
pd_ratings['rating_env'] = pd_ratings['rating_env'].fillna(rounded_average_rating_env)

# 计算rating_flavor属性的平均值
average_rating_flavor = pd_ratings['rating_flavor'].mean()
rounded_average_rating_flavor = round(average_rating_flavor, 1)
pd_ratings['rating_flavor'] = pd_ratings['rating_flavor'].fillna(rounded_average_rating_flavor)

# 计算rating_service属性的平均值
average_rating_service = pd_ratings['rating_service'].mean()
rounded_average_rating_service = round(average_rating_service, 1)
pd_ratings['rating_service'] = pd_ratings['rating_service'].fillna(rounded_average_rating_service)

# 去除较短的文本
pd_ratings['comment'] = pd_ratings['comment'].apply(lambda x: check_length(x))
pd_ratings =  pd_ratings.dropna(subset=['comment'])

# 大于3.5分的被定义为好评(1),否则是差评(0),并且将rating_encoding列插入到rating列后面
pd_ratings['rating'] = pd_ratings['rating'].apply(lambda x: ratings_encoder(x))

# 清洗评论
pd_ratings['comment'] = pd_ratings['comment'].apply(lambda x: clean_text(x))

# 使用 Count Vectorrizer
count_vectorizer = CountVectorizer(max_features=500)
count_matrix = count_vectorizer.fit_transform(pd_ratings['comment'])
count_feature_names = count_vectorizer.get_feature_names_out()

# 使用 TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=500)
tfidf_matrix = tfidf_vectorizer.fit_transform(pd_ratings['comment'])
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

# 训练集与测试集的划分
X_train_count, X_test_count, y_train_count, y_test_count = train_test_split(
    count_matrix,  
    pd_ratings['rating'],  
    test_size=0.2,  
    random_state=77  
)

X_train_count_df = pd.DataFrame(X_train_count.toarray(), columns=count_feature_names)
X_test_count_df = pd.DataFrame(X_test_count.toarray(), columns=count_feature_names)
X_train_count_df['rating'] = y_train_count.reset_index(drop=True)
X_test_count_df['rating'] = y_test_count.reset_index(drop=True)

X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(
    tfidf_matrix,  
    pd_ratings['rating'],  
    test_size=0.2,  
    random_state=77  
)

X_train_tfidf_df = pd.DataFrame(X_train_tfidf.toarray(), columns=tfidf_feature_names)
X_test_tfidf_df = pd.DataFrame(X_test_tfidf.toarray(), columns=tfidf_feature_names)
X_train_tfidf_df['rating'] = y_train_tfidf.reset_index(drop=True)
X_test_tfidf_df['rating'] = y_test_tfidf.reset_index(drop=True)

X_train_count_df.to_csv('./datasets/train_count_set.csv', index=False, encoding='utf-8')
X_test_count_df.to_csv('./datasets/test_count_set.csv', index=False, encoding='utf-8')
X_train_tfidf_df.to_csv('./datasets/train_tfidf_set.csv', index=False, encoding='utf-8')
X_test_tfidf_df.to_csv('./datasets/test_tfidf_set.csv', index=False, encoding='utf-8')
