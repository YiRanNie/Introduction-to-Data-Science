import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from math import pi

# 加载数据集
restaurants_df = pd.read_csv('datasets/restaurants.csv')
reviews_df = pd.read_csv('datasets/original_data.csv')

plt.rcParams['font.sans-serif'] = ['kaiti']
plt.rcParams["axes.unicode_minus"] = False  # 解决图像中的"-"负号的乱码问题

# 合并数据集
merged_df = pd.merge(reviews_df, restaurants_df, on='restId', how='left')

def query_restaurants(keyword):
    # 使用关键词查询餐馆名称
    results = merged_df[merged_df['name'].str.contains(keyword, case=False, na=False)]
    return results

def generate_word_clouds(restaurant_name, save=False, file_name_prefix='good_comments_restaurants_wordcloud'):
    # 筛选特定餐馆的所有评论
    restaurant_reviews = merged_df[merged_df['name'] == restaurant_name]['comment']
    
    # 合并所有评论为长字符串
    all_text = ' '.join(restaurant_reviews.dropna().astype(str))
    
    # 根据评分分为好评和差评
    good_reviews = merged_df[(merged_df['name'] == restaurant_name) & (merged_df['rating'] == 1)]['comment']
    bad_reviews = merged_df[(merged_df['name'] == restaurant_name) & (merged_df['rating'] == 0)]['comment']
    
    # 合并好评和差评为长字符串
    good_text = ' '.join(good_reviews.dropna().astype(str))
    bad_text = ' '.join(bad_reviews.dropna().astype(str))
    
    # 设置停用词列表
    stopwords = set(['我', '都', '就', '感觉', '一个', '在', '是', '的', '和', '了', '我们', '有', '也', '不', '人', '很', '到', '说', '去', '但是', '因为', '这个', '可以', '他们', '她', '自己', '这', '会', '你', '他', '时间', '来', '用', '对', '好的', '好', '如果', '工作', '吧', '没有', '我们', '这样', '这里', '那样', '或者', '就是', '吧', '吧'])

    # 创建词云对象
    wordcloud_all = WordCloud(
        font_path="DENG.TTF",
        width=1980,
        height=1080,
        background_color="white",
        stopwords=stopwords
    ).generate(all_text)
    
    wordcloud_good = WordCloud(
        font_path="DENG.TTF",
        width=1980,
        height=1080,
        background_color="white",
        stopwords=stopwords
    ).generate(good_text)
    
    wordcloud_bad = WordCloud(
        font_path="DENG.TTF",
        width=1980,
        height=1080,
        background_color="white",
        stopwords=stopwords
    ).generate(bad_text)
    
    # 显示和保存所有评论的词云
    display_and_save_wordcloud(wordcloud_all, f"{restaurant_name} 所有评论", save, file_name_prefix + '_all')

    # 显示和保存好评的词云
    display_and_save_wordcloud(wordcloud_good, f"{restaurant_name} 的好评", save, file_name_prefix + '_good')

    # 显示和保存差评的词云
    display_and_save_wordcloud(wordcloud_bad, f"{restaurant_name} 的差评", save, file_name_prefix + '_bad')

def display_and_save_wordcloud(wordcloud, title, save, file_name):
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    
    if save:
        plt.savefig(file_name + '.png', bbox_inches='tight', pad_inches=0)
    
    plt.show()

def plot_review_ratio_pie_chart(restaurant_name):
    good_reviews = merged_df[(merged_df['name'] == restaurant_name) & (merged_df['rating'] == 1)]
    bad_reviews = merged_df[(merged_df['name'] == restaurant_name) & (merged_df['rating'] == 0)]
    
    sizes = [len(good_reviews), len(bad_reviews)]
    labels = ['好评', '差评']
    colors = ['#66b3ff', '#ff6666']
    explode = (0.1, 0)  # slightly explode the good reviews slice

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title(f"{restaurant_name} 饼状图")
    plt.show()

def plot_radar_chart(restaurant_name):
    ratings = merged_df[merged_df['name'] == restaurant_name][['rating_env', 'rating_flavor', 'rating_service']]
    if ratings.empty:
        print("没有找到该餐馆的评分数据。")
        return
    
    avg_ratings = ratings.mean()
    
    categories = ['环境', '口味', '服务']
    values = avg_ratings.values
    values = list(values) + [values[0]]
    
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]
    
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories)
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, 'b', alpha=0.1)
    
    plt.title(f"{restaurant_name} 评分雷达图")
    plt.show()

def plot_positive_negative_ratio_over_time(restaurant_name, interval='year'):
    restaurant_data = merged_df[merged_df['name'] == restaurant_name]
    
    restaurant_data['timestamp'] = pd.to_datetime(restaurant_data['timestamp'], unit='ms')
    
    if interval == 'year':
        restaurant_data['year'] = restaurant_data['timestamp'].dt.year
        grouped_data = restaurant_data.groupby('year')
    elif interval == 'month':
        restaurant_data['month'] = restaurant_data['timestamp'].dt.to_period('M')
        grouped_data = restaurant_data.groupby('month')
    else:
        raise ValueError("Invalid interval. Use 'year' or 'month'.")
    
    positive_count = grouped_data['rating'].apply(lambda x: (x == 1).sum())
    negative_count = grouped_data['rating'].apply(lambda x: (x == 0).sum())
    total_count = positive_count + negative_count
    
    positive_ratio = positive_count / total_count
    negative_ratio = negative_count / total_count
    
    plt.figure(figsize=(10, 6))
    plt.plot(positive_ratio.index, positive_ratio, label='好评率', color='g')
    plt.plot(negative_ratio.index, negative_ratio, label='差评率', color='r')
    plt.xlabel(f'{interval.capitalize()}')
    plt.ylabel('比率')
    plt.title(f'{restaurant_name} 好评/差评比率[按年聚集]')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    print("请输入关键词查询餐馆：")
    keyword = input()
    
    # 查询餐馆
    results = query_restaurants(keyword)
    
    if not results.empty:
        print("找到以下餐馆：")
        for index, row in results.iterrows():
            if not pd.isna(row['name']):  # 确保名称不为空
                print(f"{index + 1}. {row['name']}")
        
        # 让用户选择一个餐馆
        if len(results) > 0:
            choice = input("请选择一个餐馆（输入编号）：")
            try:
                selected_index = int(choice) - 1
                selected_restaurant = results.iloc[selected_index]
                print(f"您选择的餐馆是：{selected_restaurant['name']}")
                
                print("是否生成并保存词云图像？(yes/no): ")
                save = input().lower() == 'yes'
                
                generate_word_clouds(selected_restaurant['name'], save)
                plot_review_ratio_pie_chart(selected_restaurant['name'])
                plot_radar_chart(selected_restaurant['name'])
                plot_positive_negative_ratio_over_time(selected_restaurant['name'], interval='year')
            except (ValueError, IndexError):
                print("输入错误，请输入有效的编号。")
        else:
            print("没有找到匹配的餐馆。")
    else:
        print("没有找到匹配的餐馆。")

if __name__ == "__main__":
    main()