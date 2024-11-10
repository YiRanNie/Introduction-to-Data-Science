import pandas as pd

train_count_set = pd.read_csv('datasets/ratings.csv')
print(train_count_set.head(50))


# 数据格式示例
# word1-wordn是样本特征
# rating是label

# num word1 word2 word3 ... rating
# 0   0     0     1         1
# 1   1     0     1         0
# 0   0     0     0         0