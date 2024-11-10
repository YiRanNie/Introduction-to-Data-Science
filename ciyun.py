import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 读取数据集
# 假设数据集已加载为 DataFrame df
# 你可以使用 df = pd.read_csv('your_file.csv') 来加载数据
df = pd.read_csv('datasets/train_count_set_min.csv') 

# 移除非特征列
df_features = df.drop(columns=['rating'])

# 计算每个特征词的出现总和
word_sums = df_features.sum(axis=0)
print(word_sums)

# 按降序排列，选择前80个高频词
top_80_words = word_sums.nlargest(150)
print(top_80_words)

# 将高频词转换为字典格式，便于词云生成
word_freq_dict = top_80_words.to_dict()

# 生成词云
wordcloud = WordCloud(
    font_path="DENG.TTF",  # 这里替换为下载的 simhei.ttf 的路径
    width=1980,
    height=1080,
    background_color="white"
).generate_from_frequencies(word_freq_dict)

# 显示词云
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
