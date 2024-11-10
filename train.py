import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

print("开始加载数据...")
# 加载训练数据集
train_df = pd.read_csv('datasets/train_tfidf_set_min.csv')

print("准备特征和标签...")
# 准备特征和标签
X_train = train_df.drop(columns=['rating'])
y_train = train_df['rating']

# 训练不同的模型
print("训练逻辑回归模型...")
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train, y_train)

print("训练决策树模型...")
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)

print("训练SVM模型...")
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

print("训练KNN模型...")
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# 保存模型
import joblib
joblib.dump(logistic_model, 'models/logistic_model.pkl')
joblib.dump(decision_tree_model, 'models/decision_tree_model.pkl')
joblib.dump(svm_model, 'models/svm_model.pkl')
joblib.dump(knn_model, 'models/knn_model.pkl')

print("模型训练完成并保存。")