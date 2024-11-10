import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

print("开始加载数据...")
# 加载测试数据集
test_df = pd.read_csv('datasets/test_tfidf_set_min.csv')

print("准备特征和标签...")
# 准备特征和标签
X_test = test_df.drop(columns=['rating'])
y_test = test_df['rating']

# 加载模型
logistic_model = joblib.load('models/logistic_model.pkl')
decision_tree_model = joblib.load('models/decision_tree_model.pkl')
svm_model = joblib.load('models/svm_model.pkl')
knn_model = joblib.load('models/knn_model.pkl')

# 预测并计算准确率
print("预测并计算准确率...")
y_pred_logistic = logistic_model.predict(X_test)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print("逻辑回归模型在测试集上的准确率:", accuracy_logistic)

y_pred_tree = decision_tree_model.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print("决策树模型在测试集上的准确率:", accuracy_tree)

y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM模型在测试集上的准确率:", accuracy_svm)

y_pred_knn = knn_model.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("KNN模型在测试集上的准确率:", accuracy_knn)