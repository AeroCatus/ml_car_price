# -*- coding: utf-8 -*-
"""
二手车价格预测 - LightGBM模型
"""

from config import Paths
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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
    sale_ids = joblib.load(str(Paths.Features.fe_sale_ids))
    return x_train, x_val, y_train, y_val, x_test, sale_ids


def train_lightgbm_model(x_train, x_val, y_train, y_val):
    """
    训练LightGBM模型
    """
    print("正在训练LightGBM模型...")

    # 创建数据集
    train_data = lgb.Dataset(x_train, label=y_train)
    val_data = lgb.Dataset(x_val, label=y_val, reference=train_data)

    # 设置模型参数
    params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.01,  # 学习率
        "num_boost_round": 20000,  # 最大迭代次数
        "num_leaves": 55,  # 叶子节点数
        "max_depth": 11,
        "min_data_in_leaf": 6,  # 每个叶子节点最少样本数
        "feature_fraction": 0.6,  # 相当于XGBoost的colsample_bytree
        "bagging_fraction": 0.6,  # 相当于XGBoost的subsample
        "bagging_freq": 1,  # 每1次迭代执行一次bagging
        "lambda_l1": 0.02,  # L1正则化
        "lambda_l2": 0.02,  # L2正则化
        "min_gain_to_split": 0.003,  # 节点分裂所需的最小增益
        "path_smooth": 0.15,  # 路径平滑参数
        "verbose": 1,
    }

    print("\n开始训练...")
    # 训练模型
    callbacks = [
        lgb.early_stopping(stopping_rounds=100),  # 早停
        lgb.log_evaluation(period=100),  # 每100轮打印一次评估结果
    ]

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=["训练集", "验证集"],
        callbacks=callbacks,
    )

    # 保存模型
    model.save_model(
        str(Paths.Models.lightgbm / "lightgbm_model.txt"),
        num_iteration=model.best_iteration,
    )
    print("\n模型已保存到 lightgbm_model.txt")

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
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "r--", lw=2)
    plt.xlabel("实际价格")
    plt.ylabel("预测价格")
    plt.title("预测价格 vs 实际价格")
    plt.tight_layout()
    plt.savefig(str(Paths.Results.plots / "lightgbm_prediction_vs_actual.png"))
    plt.close()

    return rmse, mae, r2


def plot_feature_importance(model, x_train):
    """
    绘制特征重要性图
    """
    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame(
        {"feature": x_train.columns, "importance": model.feature_importance("gain")}
    )
    importance_df = importance_df.sort_values("importance", ascending=False)

    # 保存特征重要性到CSV文件
    importance_df.to_csv(
        str(Paths.Results.importance / "lightgbm_feature_importance.csv"), index=False
    )

    # 绘制前20个最重要的特征
    plt.figure(figsize=(12, 6))
    sns.barplot(x="importance", y="feature", data=importance_df.head(20))
    plt.title("LightGBM - Top 20 特征重要性")
    plt.tight_layout()
    plt.savefig(str(Paths.Results.importance / "lightgbm_feature_importance.png"))
    plt.close()


def predict_test_data(model, test_data, sale_ids):
    """
    预测测试集数据
    """

    # 预测
    print("正在预测测试集...")
    predictions = model.predict(test_data)

    # 创建提交文件
    submit_data = pd.DataFrame({"SaleID": sale_ids, "price": predictions})

    # 保存预测结果
    submit_data.to_csv(str(Paths.Results.submission / "lightgbm_submit_result.csv"), index=False)
    print("预测结果已保存到 lightgbm_submit_result.csv")


def main():
    # 加载预处理后的数据
    x_train, x_val, y_train, y_val, test_data, sale_ids = load_processed_data()

    # 训练模型
    model = train_lightgbm_model(x_train, x_val, y_train, y_val)

    # 加载模型
    model = lgb.Booster(model_file=str(Paths.Models.lightgbm / "lightgbm_model.txt"))

    # 评估模型
    evaluate_model(model, x_val, y_val)

    # 绘制特征重要性
    plot_feature_importance(model, x_train)

    # 预测测试集
    predict_test_data(model, test_data, sale_ids)

    print("\n模型训练、评估和预测完成！")


if __name__ == "__main__":
    main()
