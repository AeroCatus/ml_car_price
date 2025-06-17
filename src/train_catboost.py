# -*- coding: utf-8 -*-
"""
二手车价格预测 - CatBoost模型
"""

from config import Paths
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings

warnings.filterwarnings("ignore")


# 设置中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


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


def train_catboost_model(x_train, x_val, y_train, y_val, cat_features):
    """
    训练CatBoost模型
    """
    print("\n开始训练CatBoost模型...")

    # 创建数据池
    train_pool = Pool(x_train, y_train, cat_features=cat_features)
    val_pool = Pool(x_val, y_val, cat_features=cat_features)

    # 设置模型参数
    params = {
        "iterations": 15000,  # 最大迭代次数
        "learning_rate": 0.01,  # 学习率
        "depth": 6,  # 树的深度
        "l2_leaf_reg": 3,  # L2正则化
        "bootstrap_type": "Bayesian",  # 采样方式
        "random_seed": 42,  # 随机种子
        "od_type": "Iter",  # 早停类型
        "od_wait": 100,  # 早停等待轮数
        "verbose": 100,  # 每100轮打印一次
        "loss_function": "MAE",  # 损失函数
        "eval_metric": "MAE",  # 评估指标
        "task_type": "CPU",  # 使用CPU训练
        "thread_count": -1,  # 使用所有CPU核心
    }

    # 创建模型
    model = CatBoostRegressor(**params)

    # 训练模型
    model.fit(train_pool, eval_set=val_pool, use_best_model=True, plot=True)

    # 保存模型
    model.save_model(str(Paths.Models.catboost / "fe_catboost_model.cbm"), format="cbm")
    print("模型已保存到 fe_catboost_model.cbm")

    return model


def evaluate_model(model, x_val, y_val, cat_features):
    """
    评估模型性能
    """
    # 创建验证数据池
    val_pool = Pool(x_val, cat_features=cat_features)

    # 预测
    y_pred = model.predict(val_pool)

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
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "r--", lw=2)
    plt.xlabel("实际价格")
    plt.ylabel("预测价格")
    plt.title("CatBoost预测价格 vs 实际价格")
    plt.tight_layout()
    plt.savefig(str(Paths.Results.plots / "fe_catboost_prediction_vs_actual.png"))
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
        str(Paths.Results.importance / "fe_catboost_feature_importance.csv"),
        index=False,
    )

    # 绘制特征重要性图
    plt.figure(figsize=(14, 8))
    sns.barplot(x="importance", y="feature", data=importance_df.head(20))
    plt.title("CatBoost Top 20 特征重要性")
    plt.tight_layout()
    plt.savefig(str(Paths.Results.importance / "fe_catboost_feature_importance.png"))
    plt.close()

    return importance_df


def predict_test_data(model, x_test, test_ids, cat_features):
    """
    预测测试集数据
    """
    print("\n正在预测测试集...")

    # 创建测试数据池
    test_pool = Pool(x_test, cat_features=cat_features)

    # 预测
    predictions = model.predict(test_pool)

    # 创建提交文件
    submit_data = pd.DataFrame({"SaleID": test_ids, "price": predictions})

    # 保存预测结果
    submit_data.to_csv(
        str(Paths.Results.submission / "fe_catboost_submit_result.csv"), index=False
    )
    print("预测结果已保存到 fe_catboost_submit_result.csv")


def main():
    # 加载预处理后的数据
    x_train, x_val, y_train, y_val, x_test, test_ids, cat_features = (
        load_processed_data()
    )

    # 训练CatBoost模型
    model = train_catboost_model(x_train, x_val, y_train, y_val, cat_features)

    # 加载模型
    model = CatBoostRegressor()
    model.load_model(str(Paths.Models.catboost / "fe_catboost_model.cbm"))

    # 评估模型
    evaluate_model(model, x_val, y_val, cat_features)

    # 绘制特征重要性
    plot_feature_importance(model, x_train)

    # 预测测试集
    predict_test_data(model, x_test, test_ids, cat_features)

    print("\n模型训练、评估和预测完成！")


if __name__ == "__main__":
    main()
