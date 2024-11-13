import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据集
df = pd.read_csv('D:\Data Science\datasets\original_data.csv', nrows=10000)

# 计算每个评论的评分总和
df['total_rating'] = df[['rating', 'rating_env', 'rating_flavor', 'rating_service']].sum(axis=1)

# 筛选出前150个好评的评论
top_reviews = df.nlargest(150, 'total_rating')

# 提取这些评论的文本
top_comments = top_reviews['comment'].tolist()

# 将所有好评评论合并成一个大字符串
all_top_comments = ' '.join(top_comments)

# 设置停用词列表
stopwords = set(['我', '都', '就', '感觉', '一个', '在', '是', '的', '和', '了', '我们', '有', '也', '不', '人', '很', '到', '说', '去', '但是', '因为', '这个', '可以', '他们', '她', '自己', '这', '会', '你', '他', '时间', '来', '用', '对', '好的', '好', '如果', '工作', '吧', '没有', '我们', '这样', '这里', '那样', '或者', '吧', '吧', '吧'])

# 创建好评词云对象，设置中文字体路径
wordcloud_top = WordCloud(
    font_path="DENG.TTF",  # 这里替换为下载的 simhei.ttf 的路径
    width=1980,
    height=1080,
    background_color="white",
    stopwords=stopwords
).generate(all_top_comments)

# 显示好评词云图像
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_top, interpolation='bilinear')
plt.axis('off')
plt.title('Top Reviews Word Cloud')
plt.show()

# 如果需要，可以将好评词云保存为文件
wordcloud_top.to_file('positive_wordcloud.png')

# 筛选出rating为1的评论
# 这里我们假设差评是指rating为1的评论
negative_reviews = df[df['rating'] == 1]['comment']

# 提取这些评论的文本
negative_comments = negative_reviews.tolist()

# 将所有差评评论合并成一个大字符串
all_negative_comments = ' '.join(negative_comments)

# 创建差评词云对象
wordcloud_negative = WordCloud(
    font_path="DENG.TTF",  # 这里替换为下载的 simhei.ttf 的路径
    width=1980,
    height=1080,
    background_color="white",
    stopwords=stopwords
).generate(all_negative_comments)

# 显示差评词云图像
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis('off')
plt.title('Negative Reviews Word Cloud')
plt.show()

# 如果需要，可以将差评词云保存为文件
wordcloud_negative.to_file('negative_wordcloud.png')

# 提取所有评论的文本
all_comments = df['comment'].tolist()

# 将所有评论合并成一个大字符串
all_comments_str = ' '.join(all_comments)

# 创建所有评论的词云对象
wordcloud_all = WordCloud(
    font_path="DENG.TTF",  # 这里替换为下载的 simhei.ttf 的路径
    width=1980,
    height=1080,
    background_color="white",
    stopwords=stopwords
).generate(all_comments_str)

# 显示所有评论的词云图像
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_all, interpolation='bilinear')
plt.axis('off')
plt.title('All Reviews Word Cloud')
plt.show()

# 如果需要，可以将所有评论的词云保存为文件
wordcloud_all.to_file('all_wordcloud.png')

# 读取数据
original_data = pd.read_csv('D:\Data Science\datasets\original_data.csv')
restaurants_data = pd.read_csv('D:\Data Science\datasets/restaurants.csv')

# 合并数据
merged_data = pd.merge(original_data, restaurants_data, on='restId', how='inner')

# 过滤掉没有餐馆名称的记录
merged_data = merged_data.dropna(subset=['name'])

# 确保rating为0和1的数据都有正确计数
# 按name分组，统计rating为0和1的数量
restaurant_ratings = merged_data.groupby(['name', 'rating']).size().unstack(fill_value=0)
restaurant_ratings.columns = ['差评数', '好评数']

# 增加一个列，计算总评价次数
restaurant_ratings['总评价数'] = restaurant_ratings['好评数'] + restaurant_ratings['差评数']

# 过滤只保留总评价数 > 300 的餐馆
restaurant_ratings = restaurant_ratings[restaurant_ratings['总评价数'] > 300]

# 计算好评率和差评率
restaurant_ratings['好评率'] = restaurant_ratings['好评数'] / restaurant_ratings['总评价数']
restaurant_ratings['差评率'] = restaurant_ratings['差评数'] / restaurant_ratings['总评价数']

# 排序并选择好评率最高的前10家餐馆
top10_restaurants = restaurant_ratings.sort_values(by='好评率', ascending=False).head(10)
top10_names = top10_restaurants.index  # 获取Top 10餐馆的名称列表

top10_restaurants[['好评率', '差评率']].plot(kind='bar', stacked=True, figsize=(10, 6), color=['skyblue', 'salmon'])
plt.title("Top 10 餐馆的好评/差评比率（总评价数 > 300）")
plt.xlabel("餐馆名称")
plt.ylabel("比率")
plt.xticks(rotation=45, ha="right")
plt.legend(title="评分类型")
plt.tight_layout()
plt.savefig("top10_restaurants_rating_ratio.png")  # 保存图像
plt.show()

# 转换时间戳为日期格式
merged_data['timestamp'] = pd.to_datetime(merged_data['timestamp'], unit='ms')

# 只保留Top 10餐馆的数据
top10_data = merged_data[merged_data['name'].isin(top10_names)]

# 按年进行聚合，计算每年的好评和差评数量
top10_data['year'] = top10_data['timestamp'].dt.to_period('Y')
yearly_ratings = top10_data.groupby(['name', 'year', 'rating']).size().unstack(fill_value=0)
yearly_ratings.columns = ['差评数', '好评数']

# 计算每年的好评率
yearly_ratings['好评率'] = yearly_ratings['好评数'] / (yearly_ratings['好评数'] + yearly_ratings['差评数'])
yearly_ratings = yearly_ratings.reset_index()

# 绘制每个餐馆的好评率随时间变化的线图（按年）
plt.figure(figsize=(12, 8))
for name in top10_names:
    restaurant_yearly = yearly_ratings[yearly_ratings['name'] == name]
    plt.plot(restaurant_yearly['year'].dt.to_timestamp(), restaurant_yearly['好评率'], label=name)

plt.title("Top 10 餐馆随时间变化的好评率（按年聚合）")
plt.xlabel("时间")
plt.ylabel("好评率")
plt.legend(title="餐馆名称", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("top10_restaurants_yearly_rating_trend.png")  # 保存图像
plt.show()

# 计算环境评分、口味评分和服务评分的平均值
mean_scores = top10_data.groupby('name')[['rating_env', 'rating_flavor', 'rating_service']].mean()

# 设置雷达图参数
labels = mean_scores.columns
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for name, row in mean_scores.iterrows():
    values = row.tolist()
    values += values[:1]  # 将数据封闭，以形成完整的多边形
    ax.plot(angles, values, label=name, linewidth=1, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)

# 设置雷达图的标题和标签
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
plt.title("Top 10 餐馆的环境、口味、服务评分雷达图")
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig("top10_restaurants_radar_chart.png") 
plt.show()