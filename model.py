import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re
from scipy.sparse import csr_matrix
import dask.dataframe as dd

# 定义 clean_text 函数
def clean_text(text):
    # 示例清洗逻辑
    text = text.lower()  # 转换为小写
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # 移除非字母数字字符
    return text

# 定义 ratings_encoder 函数
def ratings_encoder(rating):
    if rating >= 3.5:
        return 1  # 好评
    else:
        return 0  # 差评

# 训练二值分类模型（这里以逻辑回归为例）
def train_model():
    # 使用 Dask 分批次读取训练集
    X_train_count_df = dd.read_csv('./datasets/train_count_set.csv')
    
    # 打印数据集的列名，确认是否包含 'comment' 列
    print("训练集列名:", X_train_count_df.columns.tolist())

    # 如果列名不正确，可以进行列名重命名
    if 'comment' not in X_train_count_df.columns:
        raise ValueError("训练集中没有 'comment' 列，请检查数据文件或列名拼写。")

    count_vectorizer = CountVectorizer(max_features=200)
    X_train_count_sparse = count_vectorizer.fit_transform(X_train_count_df['comment'].compute())
    # 将稀疏矩阵转换为 CSR 矩阵
    X_train_count_csr = csr_matrix(X_train_count_sparse)

    lr_model = LogisticRegression()
    lr_model.fit(X_train_count_csr, X_train_count_df['rating'].compute())
    return lr_model, count_vectorizer

# 在测试集上计算 accuracy
def test_accuracy(lr_model, count_vectorizer):
    # 使用 Dask 分批次读取测试集
    X_test_count_df = dd.read_csv('./datasets/test_count_set.csv')
    
    # 打印数据集的列名，确认是否包含 'comment' 列
    print("测试集列名:", X_test_count_df.columns.tolist())

    # 如果列名不正确，可以进行列名重命名
    if 'comment' not in X_test_count_df.columns:
        raise ValueError("测试集中没有 'comment' 列，请检查数据文件或列名拼写。")

    X_test_count_sparse = count_vectorizer.transform(X_test_count_df['comment'].compute())
    X_test_count_csr = csr_matrix(X_test_count_sparse)
    y_pred = lr_model.predict(X_test_count_csr)
    accuracy = accuracy_score(X_test_count_df['rating'].compute(), y_pred)
    print(f"测试集准确率：{accuracy}")

# 用户输入新的评论得到好评/差评分类结果
def classify_new_comment(lr_model, count_vectorizer):
    new_comment = input("请输入新的评论：")
    cleaned_new_comment = clean_text(new_comment)
    new_comment_vectorized = count_vectorizer.transform([cleaned_new_comment])
    new_comment_vectorized_csr = csr_matrix(new_comment_vectorized)
    new_prediction = lr_model.predict(new_comment_vectorized_csr)[0]
    if new_prediction == 1:
        print("好评")
    else:
        print("差评")

# all 评论的高频词/词云可视化
def visualize_all_wordclouds():
    pd_ratings = dd.read_csv('./datasets/ratings.csv')
    
    # 打印数据集的列名，确认是否包含 'comment' 列
    print("评论数据集列名:", pd_ratings.columns.tolist())

    # 如果列名不正确，可以进行列名重命名
    if 'comment' not in pd_ratings.columns:
        raise ValueError("评论数据集中没有 'comment' 列，请检查数据文件或列名拼写。")

    pd_ratings = pd_ratings.drop_duplicates()
    pd_ratings = pd_ratings.dropna(subset=['comment'])
    rounded_average_rating = pd_ratings['rating'].mean().round()
    rounded_average_rating_env = pd_ratings['rating_env'].mean().round()
    rounded_average_rating_flavor = pd_ratings['rating_flavor'].mean().round()
    rounded_average_rating_service = pd_ratings['rating_service'].mean().round()
    pd_ratings['rating'] = pd_ratings['rating'].fillna(rounded_average_rating)
    pd_ratings['rating_env'] = pd_ratings['rating_env'].fillna(rounded_average_rating_env)
    pd_ratings['rating_flavor'] = pd_ratings['rating_flavor'].fillna(rounded_average_rating_flavor)
    pd_ratings['rating_service'] = pd_ratings['rating_service'].fillna(rounded_average_rating_service)
    pd_ratings = pd_ratings.dropna(subset=['comment'])
    pd_ratings['rating'] = pd_ratings['rating'].apply(lambda x: ratings_encoder(x))
    pd_ratings['comment'] = pd_ratings['comment'].apply(lambda x: clean_text(x))

    all_comments = pd_ratings['comment'].compute().values
    all_words = []
    for comment in all_comments:
        words = comment.split()
        all_words.extend(words)
    word_counts_all = Counter(all_words)
    most_common_words_all = word_counts_all.most_common(20)
    print("所有评论的高频词：", most_common_words_all)

    wordcloud_all = WordCloud(width=800, height=400).generate_from_frequencies(word_counts_all)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_all)
    plt.axis('off')
    plt.title("所有评论词云")
    plt.show()

# all 好评的高频词/词云可视化
def visualize_positive_wordclouds():
    pd_ratings = dd.read_csv('./datasets/ratings.csv')
    
    # 打印数据集的列名，确认是否包含 'comment' 列
    print("评论数据集列名:", pd_ratings.columns.tolist())

    # 如果列名不正确，可以进行列名重命名
    if 'comment' not in pd_ratings.columns:
        raise ValueError("评论数据集中没有 'comment' 列，请检查数据文件或列名拼写。")

    pd_ratings = pd_ratings.drop_duplicates()
    pd_ratings = pd_ratings.dropna(subset=['comment'])
    rounded_average_rating = pd_ratings['rating'].mean().round()
    rounded_average_rating_env = pd_ratings['rating_env'].mean().round()
    rounded_average_rating_flavor = pd_ratings['rating_flavor'].mean().round()
    rounded_average_rating_service = pd_ratings['rating_service'].mean().round()
    pd_ratings['rating'] = pd_ratings['rating'].fillna(rounded_average_rating)
    pd_ratings['rating_env'] = pd_ratings['rating_env'].fillna(rounded_average_rating_env)
    pd_ratings['rating_flavor'] = pd_ratings['rating_flavor'].fillna(rounded_average_rating_flavor)
    pd_ratings['rating_service'] = pd_ratings['rating_service'].fillna(rounded_average_rating_service)
    pd_ratings = pd_ratings.dropna(subset=['comment'])
    pd_ratings['rating'] = pd_ratings['rating'].apply(lambda x: ratings_encoder(x))
    pd_ratings['comment'] = pd_ratings['comment'].apply(lambda x: clean_text(x))

    positive_comments = pd_ratings[pd_ratings['rating'] == 1]['comment'].compute().values
    positive_words = []
    for comment in positive_comments:
        words = comment.split()
        positive_words.extend(words)
    word_counts_positive = Counter(positive_words)
    most_common_words_positive = word_counts_positive.most_common(20)
    print("所有好评的高频词：", most_common_words_positive)

    wordcloud_positive = WordCloud(width=800, height=400).generate_from_frequencies(word_counts_positive)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_positive)
    plt.axis('off')
    plt.title("所有好评词云")
    plt.show()

# all 差评的高频词/词云可视化
def visualize_negative_wordclouds():
    pd_ratings = dd.read_csv('./datasets/ratings.csv')
    
    # 打印数据集的列名，确认是否包含 'comment' 列
    print("评论数据集列名:", pd_ratings.columns.tolist())

    # 如果列名不正确，可以进行列名重命名
    if 'comment' not in pd_ratings.columns:
        raise ValueError("评论数据集中没有 'comment' 列，请检查数据文件或列名拼写。")

    pd_ratings = pd_ratings.drop_duplicates()
    pd_ratings = pd_ratings.dropna(subset=['comment'])
    rounded_average_rating = pd_ratings['rating'].mean().round()
    rounded_average_rating_env = pd_ratings['rating_env'].mean().round()
    rounded_average_rating_flavor = pd_ratings['rating_flavor'].mean().round()
    rounded_average_rating_service = pd_ratings['rating_service'].mean().round()
    pd_ratings['rating'] = pd_ratings['rating'].fillna(rounded_average_rating)
    pd_ratings['rating_env'] = pd_ratings['rating_env'].fillna(rounded_average_rating_env)
    pd_ratings['rating_flavor'] = pd_ratings['rating_flavor'].fillna(rounded_average_rating_flavor)
    pd_ratings['rating_service'] = pd_ratings['rating_service'].fillna(rounded_average_rating_service)
    pd_ratings = pd_ratings.dropna(subset=['comment'])
    pd_ratings['rating'] = pd_ratings['rating'].apply(lambda x: ratings_encoder(x))
    pd_ratings['comment'] = pd_ratings['comment'].apply(lambda x: clean_text(x))

    negative_comments = pd_ratings[pd_ratings['rating'] == 0]['comment'].compute().values
    negative_words = []
    for comment in negative_comments:
        words = comment.split()
        negative_words.extend(words)
    word_counts_negative = Counter(negative_words)
    most_common_words_negative = word_counts_negative.most_common(20)
    print("所有差评的高频词：", most_common_words_negative)

    wordcloud_negative = WordCloud(width=800, height=400).generate_from_frequencies(word_counts_negative)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_negative)
    plt.axis('off')
    plt.title("所有差评词云")
    plt.show()

# 主程序
lr_model, count_vectorizer = train_model()
test_accuracy(lr_model, count_vectorizer)
classify_new_comment(lr_model, count_vectorizer)
visualize_all_wordclouds()
visualize_positive_wordclouds()
visualize_negative_wordclouds()