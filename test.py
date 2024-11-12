import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

# 加载模型
logistic_model = joblib.load('models/logistic_model_4.pkl')
decision_tree_model = joblib.load('models/decision_tree_model_4.pkl')
svm_model = joblib.load('models/svm_model_4.pkl')
knn_model = joblib.load('models/knn_model_4.pkl')

# 定义数据集路径和名称
datasets = {
    "TF-IDF 测试数据集": 'datasets/test_tfidf_set_min.csv',
    "Count 测试数据集": 'datasets/test_count_set_min.csv'
}

# 遍历数据集
for dataset_name, dataset_path in datasets.items():
    print(f"\n开始加载 {dataset_name}...")
    
    # 加载数据集
    test_df = pd.read_csv(dataset_path)
    
    print("准备特征和标签...")
    X_test = test_df.drop(columns=['rating'])
    y_test = test_df['rating']
    
    print(f"在 {dataset_name} 上的预测并计算准确率...")
    
    # 逻辑回归模型
    y_pred_logistic = logistic_model.predict(X_test)
    accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
    print(f"{dataset_name} - 逻辑回归模型在测试集上的准确率:", accuracy_logistic)
    
    # 决策树模型
    y_pred_tree = decision_tree_model.predict(X_test)
    accuracy_tree = accuracy_score(y_test, y_pred_tree)
    print(f"{dataset_name} - 决策树模型在测试集上的准确率:", accuracy_tree)
    
    # SVM模型
    y_pred_svm = svm_model.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print(f"{dataset_name} - SVM模型在测试集上的准确率:", accuracy_svm)
    
    # KNN模型
    y_pred_knn = knn_model.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    print(f"{dataset_name} - KNN模型在测试集上的准确率:", accuracy_knn)
