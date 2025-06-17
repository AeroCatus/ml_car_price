# -*- coding: utf-8 -*-
"""
二手车价格预测 - XGBoost模型
"""

from config import Paths
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings("ignore")

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_processed_data():
    """
    加载预处理后的数据
    """
    print("正在加载预处理后的数据...")
    x_train = joblib.load(str(Paths.Features.fe_x_train))
    x_val = joblib.load(str(Paths.Features.fe_x_val))
    y_train = joblib.load(str(Paths.Features.fe_y_train))
    y_val = joblib.load(str(Paths.Features.fe_y_val))
    x_test = joblib.load(str(Paths.Features.fe_test_data))
    test_ids = joblib.load(str(Paths.Features.fe_sale_ids))
    cat_features = joblib.load(str(Paths.Features.fe_cat_features))
    return x_train, x_val, y_train, y_val, x_test, test_ids, cat_features


def train_xgboost_model(x_train, x_val, y_train, y_val):
    """
    训练XGBoost模型
    """
    for col in x_train.select_dtypes(include='category').columns:
        x_train[col] = x_train[col].cat.codes
        x_val[col] = x_val[col].cat.codes

    print("正在训练XGBoost模型...")

    # 设置模型参数
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.01,  # 降低学习率
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 8000,  # 增加树的数量
        'random_state': 42,
        'eval_metric': 'mae',
        'early_stopping_rounds': 100,  # 添加早停机制
    }

    # 创建模型
    model = xgb.XGBRegressor(**params)

    # 训练模型
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        verbose=100,  # 每100轮打印一次评估结果
    )

    # 获取最佳迭代次数
    best_iteration = model.best_iteration
    print(f"\n最佳迭代次数: {best_iteration}")

    # 保存模型
    joblib.dump(model, str(Paths.Models.xgboost / 'fe_xgboost_model.joblib'))
    print("模型已保存到 fe_xgboost_model.joblib")

    return model


def evaluate_model(model, x_val, y_val):
    """
    评估模型性能
    """
    # 预测
    y_pred = model.predict(x_val)

    # 计算评估指标
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print("\n模型评估结果：")
    print(f"均方根误差 (RMSE): {rmse:.2f}")
    print(f"平均绝对误差 (MAE): {mae:.2f}")
    print(f"R2分数: {r2:.4f}")

    # 绘制预测值与实际值的对比图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_pred, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    plt.xlabel('实际价格')
    plt.ylabel('预测价格')
    plt.title('预测价格 vs 实际价格')
    plt.tight_layout()
    plt.savefig(str(Paths.Results.plots / 'fe_xgboost_prediction_vs_actual.png'))
    plt.close()

    return rmse, mae, r2


def plot_feature_importance(model, x_train):
    """
    绘制特征重要性图
    """
    # 获取特征重要性
    feature_importance = model.get_feature_importance()
    feature_names = x_train.columns

    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": feature_importance}
    )
    importance_df = importance_df.sort_values("importance", ascending=False)

    # 保存特征重要性到CSV
    importance_df.to_csv(
        str(Paths.Results.importance / "fe_xgboost_feature_importance.csv"),
        index=False,
    )

    # 绘制特征重要性图
    plt.figure(figsize=(14, 8))
    sns.barplot(x="importance", y="feature", data=importance_df.head(20))
    plt.title("XGBoost Top 20 特征重要性")
    plt.tight_layout()
    plt.savefig(str(Paths.Results.importance / "fe_xgboost_feature_importance.png"))
    plt.close()


def predict_test_data(model, test_data, sale_ids):
    """
    预测测试集数据
    """
    # 预测
    print("正在预测测试集...")
    predictions = model.predict(test_data)

    # 创建提交文件
    submit_data = pd.DataFrame({'SaleID': sale_ids, 'price': predictions})

    # 保存预测结果
    submit_data.to_csv(
        str(Paths.Results.submission / 'fe_xgboost_submit_result.csv'), index=False
    )
    print("预测结果已保存到 fe_xgboost_submit_result.csv")


def main():
    # 加载预处理后的数据
    x_train, x_val, y_train, y_val, test_data, sale_ids, _ = load_processed_data()

    # 训练模型
    model = train_xgboost_model(x_train, x_val, y_train, y_val)

    # 评估模型
    evaluate_model(model, x_val, y_val)

    # 绘制特征重要性
    plot_feature_importance(model, x_train)

    # 预测测试集
    predict_test_data(model, test_data, sale_ids)

    print("\n模型训练、评估和预测完成！")


if __name__ == "__main__":
    main()
