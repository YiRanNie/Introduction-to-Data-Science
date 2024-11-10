import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

print("加载所有数据（用于后续任务）...")
# 加载所有数据（用于后续任务）
df = pd.concat([pd.read_csv('datasets/train_tfidf_set_min.csv'), pd.read_csv('datasets/test_tfidf_set_min.csv')])

print("生成词云可视化...")
# 词云可视化
def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

# 提取好评和差评的文本
positive_reviews = ' '.join(df[df['rating'] == 1]['review'])
negative_reviews = ' '.join(df[df['rating'] == 0]['review'])
all_reviews = ' '.join(df['review'])

generate_wordcloud(all_reviews, "所有评论的词云")
generate_wordcloud(positive_reviews, "好评的词云")
generate_wordcloud(negative_reviews, "差评的词云")

print("计算不同餐馆的好评/差评比率和可视化...")
# 不同餐馆的好评/差评比率和可视化
restaurant_rating = df.groupby('restaurant_name')['rating'].value_counts(normalize=True).unstack().fillna(0)
restaurant_rating.columns = ['差评比率', '好评比率']
top10_restaurants = restaurant_rating.sort_values('好评比率', ascending=False).head(10)
top10_restaurants[['好评比率']].plot(kind='bar', title="Top10 餐馆的好评比率", legend=False)
plt.ylabel("好评比率")
plt.show()

print("按周聚合好评/差评比率...")
# 按周聚合好评/差评比率
df['review_date'] = pd.to_datetime(df['review_date'])
weekly_ratings = df.set_index('review_date').groupby([pd.Grouper(freq='W'), 'restaurant_name'])['rating'].mean().unstack()
top10_weekly = weekly_ratings[top10_restaurants.index]
top10_weekly.plot(title="Top10 餐馆的随时间变化的好评比率")
plt.ylabel("好评比率")
plt.show()

print("生成Top10 餐馆的环境、口味、服务评分雷达图...")
# Top10 餐馆的环境、口味、服务评分雷达图
categories = ['rating_env', 'rating_flavor', 'rating_service']
data = df.groupby('restaurant_name')[categories].mean().loc[top10_restaurants.index]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
angles += angles[:1]  # 闭合图形

for i, row in data.iterrows():
    values = row.tolist() + row.tolist()[:1]
    ax.plot(angles, values, label=i)
    ax.fill(angles, values, alpha=0.1)

plt.title("Top10 餐馆的评分雷达图")
plt.legend(loc='upper right')
plt.show()

print("关键字查询和单个餐馆的可视化分析...")
# 关键字查询和单个餐馆的可视化分析
def search_restaurant(keyword):
    return df[df['restaurant_name'].str.contains(keyword) | df['review'].str.contains(keyword)][['restaurant_name', 'review']]

# 示例：获取包含关键字“湘”的餐馆
keyword = "湘"
search_results = search_restaurant(keyword)
print("搜索结果：")
print(search_results)

print("生成餐馆评论的词云...")
# 餐馆评论的词云
restaurant_name = "选择的餐馆名称"  # 将此处替换为具体餐馆名称
restaurant_reviews = ' '.join(df[df['restaurant_name'] == restaurant_name]['review'])
generate_wordcloud(restaurant_reviews, f"{restaurant_name}的评论词云")

print("计算餐馆的好评/差评比率...")
# 餐馆的好评/差评比率
restaurant_rating = df[df['restaurant_name'] == restaurant_name]['rating'].value_counts(normalize=True)
restaurant_rating.plot(kind='pie', labels=['差评', '好评'], autopct='%1.1f%%')
plt.title(f"{restaurant_name}的好评/差评比率")
plt.show()

print("生成餐馆环境、口味、服务评分的雷达图...")
# 餐馆环境、口味、服务评分的雷达图
restaurant_scores = df[df['restaurant_name'] == restaurant_name][categories].mean()
values = restaurant_scores.tolist() + restaurant_scores.tolist()[:1]
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.plot(angles, values, linewidth=2, linestyle='solid')
ax.fill(angles, values, alpha=0.4)
plt.title(f"{restaurant_name}的评分雷达图")
plt.show()

print("生成餐馆随时间变化的好评/差评比率...")
# 餐馆随时间变化的好评/差评比率
restaurant_weekly = df[df['restaurant_name'] == restaurant_name].set_index('review_date').resample('W')['rating'].mean()
restaurant_weekly.plot(title=f"{restaurant_name}随时间变化的好评比率")
plt.ylabel("好评比率")
plt.show()