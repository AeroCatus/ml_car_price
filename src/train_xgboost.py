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
    return x_train, x_val, y_train, y_val, x_test, test_ids


def train_xgboost_model(x_train, x_val, y_train, y_val):
    """
    训练XGBoost模型
    """
    for col in x_train.select_dtypes(include="category").columns:
        x_train[col] = x_train[col].cat.codes
        x_val[col] = x_val[col].cat.codes

    print("正在训练XGBoost模型...")

    # 设置模型参数
    params = {
        # 基础参数
        "objective": "reg:squarederror",  # 回归任务
        "eval_metric": "mae",  # 评估指标：MAE
        "random_state": 42,  # 随机种子，保证结果可复现
        # 学习率与树的数量（追求极致准确率）
        "learning_rate": 0.005,  # 学习率
        "n_estimators": 20000,  # 树
        # 树的复杂度控制
        "max_depth": 8,  # 树的深度
        "min_child_weight": 5,  # 叶子节点的权重
        "gamma": 0.01,  # 节点分裂所需的最小损失
        # 正则化参数
        "subsample": 0.85,  # 行采样比例
        "colsample_bytree": 0.85,  # 列采样比例
        "reg_alpha": 0.05,  # L1正则化
        "reg_lambda": 0.1,  # L2正则化
        # 早停与迭代控制
        "early_stopping_rounds": 200,  # 早停轮数
        "verbosity": 2,  # 日志级别 0-静默，1-警告，2-信息，3-调试
        # 其他增强参数（如果有GPU，可启用加速）
        "tree_method": "hist",  # 使用直方图算法
        "gpu_id": 0,  # 启用GPU训练
        "predictor": "gpu_predictor",  # 启用GPU预测
        "verbose": 100,  # 每100轮打印一次评估结果
    }

    # 创建模型
    model = xgb.XGBRegressor(**params)

    # 训练模型
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
    )

    # 获取最佳迭代次数
    best_iteration = model.best_iteration
    print(f"\n最佳迭代次数: {best_iteration}")

    # 保存模型
    joblib.dump(model, str(Paths.Models.xgboost / "xgboost_model.joblib"))
    print("模型已保存到 xgboost_model.joblib")

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
    plt.savefig(str(Paths.Results.plots / "xgboost_prediction_vs_actual.png"))
    plt.close()

    return rmse, mae, r2


def plot_feature_importance(model, x_train):
    """
    绘制特征重要性图
    """
    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame(
        {"feature": x_train.columns, "importance": model.feature_importances_}
    )
    importance_df = importance_df.sort_values("importance", ascending=False)

    # 保存特征重要性到CSV
    importance_df.to_csv(
        str(Paths.Results.importance / "xgboost_feature_importance.csv"),
        index=False,
    )

    # 绘制特征重要性图
    plt.figure(figsize=(14, 8))
    sns.barplot(x="importance", y="feature", data=importance_df.head(20))
    plt.title("XGBoost Top 20 特征重要性")
    plt.tight_layout()
    plt.savefig(str(Paths.Results.importance / "xgboost_feature_importance.png"))
    plt.close()


def predict_test_data(model, test_data, sale_ids):
    """
    预测测试集数据
    """
    print("正在预测测试集...")

    # 预测
    predictions = model.predict(test_data)

    # 创建提交文件
    submit_data = pd.DataFrame({"SaleID": sale_ids, "price": predictions})

    # 保存预测结果
    submit_data.to_csv(
        str(Paths.Results.submission / "xgboost_submit_result.csv"), index=False
    )
    print("预测结果已保存到 xgboost_submit_result.csv")


def main():
    # 加载预处理后的数据
    x_train, x_val, y_train, y_val, test_data, sale_ids = load_processed_data()

    # 训练模型
    model = train_xgboost_model(x_train, x_val, y_train, y_val)

    # 加载模型
    model = joblib.load(str(Paths.Models.xgboost / "xgboost_model.joblib"))

    # 评估模型
    evaluate_model(model, x_val, y_val)

    # 绘制特征重要性
    plot_feature_importance(model, x_train)

    # 预测测试集
    predict_test_data(model, test_data, sale_ids)

    print("\n模型训练、评估和预测完成！")


if __name__ == "__main__":
    main()
