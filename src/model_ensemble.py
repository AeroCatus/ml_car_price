# -*- coding: utf-8 -*-
"""
模型融合 - XGBoost和LightGBM预测结果加权平均
"""

from config import Paths
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_predictions():
    """
    加载两个模型的预测结果
    """
    print("正在加载模型预测结果...")

    # 加载XGBoost预测结果
    model_1_pred = pd.read_csv(
        str(Paths.Results.submission / "xgboost_submit_result.csv")
    )
    print(f"XGBoost预测结果形状: {model_1_pred.shape}")

    # 加载CatBoost预测结果
    model_2_pred = pd.read_csv(
        str(Paths.Results.submission / "catboost_submit_result.csv")
    )
    print(f"CatBoost预测结果形状: {model_2_pred.shape}")

    # 加载LightGBM预测结果
    model_3_pred = pd.read_csv(
        str(Paths.Results.submission / "lightgbm_submit_result.csv")
    )
    print(f"LightGBM预测结果形状: {model_3_pred.shape}")

    # 验证SaleID是否一致
    if not (model_1_pred["SaleID"] == model_2_pred["SaleID"]).all():
        raise ValueError("两个模型的SaleID不一致！")
    if not (model_1_pred["SaleID"] == model_3_pred["SaleID"]).all():
        raise ValueError("两个模型的SaleID不一致！")

    return model_1_pred, model_2_pred, model_3_pred


def ensemble_predictions(
    model_1_pred, model_2_pred, model_3_pred, weights=(0.333334, 0.333333, 0.333333)
):
    """
    对三个模型的预测结果进行加权平均
    """
    print("\n开始模型融合...")
    print(f"XGBoost权重: {weights[0]}")
    print(f"CatBoost权重: {weights[1]}")
    print(f"LightGBM权重: {weights[2]}")

    # 确保权重和为1
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("权重之和必须为1！")

    # 计算加权平均
    ensemble_pred = pd.DataFrame(
        {
            "SaleID": model_1_pred["SaleID"],
            "price": weights[0] * model_1_pred["price"]
            + weights[1] * model_2_pred["price"]
            + weights[2] * model_3_pred["price"],
        }
    )

    return ensemble_pred


def analyze_predictions(model_1_pred, model_2_pred, model_3_pred, ensemble_pred):
    """
    分析三个模型及融合后的预测结果
    """
    print("\n预测结果分析：")
    print("-" * 50)

    # 基本统计信息
    print("\nXGBoost预测统计：")
    print(model_1_pred["price"].describe())
    print("\nCatBoost预测统计：")
    print(model_2_pred["price"].describe())
    print("\nLightGBM预测统计：")
    print(model_3_pred["price"].describe())
    print("\n融合后预测统计：")
    print(ensemble_pred["price"].describe())

    # 计算模型间的相关性矩阵
    corr_matrix = pd.DataFrame(
        {
            "XGBoost": model_1_pred["price"],
            "CatBoost": model_2_pred["price"],
            "LightGBM": model_3_pred["price"],
        }
    ).corr()

    print(f"\n三个模型预测结果的相关性矩阵:\n{corr_matrix.round(4)}")

    # 计算两两模型间的预测差异
    diff_1_2 = abs(model_1_pred["price"] - model_2_pred["price"])
    diff_1_3 = abs(model_1_pred["price"] - model_3_pred["price"])
    diff_2_3 = abs(model_2_pred["price"] - model_3_pred["price"])

    print("\n预测差异统计：")
    print("XGBoost vs CatBoost:")
    print(diff_1_2.describe())
    print("\nXGBoost vs LightGBM:")
    print(diff_1_3.describe())
    print("\nCatBoost vs LightGBM:")
    print(diff_2_3.describe())

    # 绘制预测结果对比图
    plt.figure(figsize=(18, 12))

    # 三模型散点图矩阵
    plt.subplot(2, 2, 1)
    sns.scatterplot(
        x=model_1_pred["price"],
        y=model_2_pred["price"],
        alpha=0.5,
        label="XGBoost vs CatBoost",
    )
    sns.scatterplot(
        x=model_1_pred["price"],
        y=model_3_pred["price"],
        alpha=0.5,
        label="XGBoost vs LightGBM",
    )
    sns.scatterplot(
        x=model_2_pred["price"],
        y=model_3_pred["price"],
        alpha=0.5,
        label="CatBoost vs LightGBM",
    )

    # 添加对角线参考线
    min_val = min(
        model_1_pred["price"].min(),
        model_2_pred["price"].min(),
        model_3_pred["price"].min(),
    )
    max_val = max(
        model_1_pred["price"].max(),
        model_2_pred["price"].max(),
        model_3_pred["price"].max(),
    )

    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    plt.xlabel("预测价格")
    plt.ylabel("预测价格")
    plt.title("三个模型预测结果对比")
    plt.legend()

    # 预测差异分布对比
    plt.subplot(2, 2, 2)
    sns.histplot(diff_1_2, bins=50, alpha=0.5, label="XGBoost vs CatBoost")
    sns.histplot(diff_1_3, bins=50, alpha=0.5, label="XGBoost vs LightGBM")
    sns.histplot(diff_2_3, bins=50, alpha=0.5, label="CatBoost vs LightGBM")
    plt.xlabel("预测差异")
    plt.ylabel("频数")
    plt.title("两两模型预测差异分布")
    plt.legend()

    # 各模型与融合结果的差异
    plt.subplot(2, 2, 3)
    sns.histplot(
        abs(model_1_pred["price"] - ensemble_pred["price"]),
        bins=50,
        alpha=0.5,
        label="XGBoost vs 融合",
    )
    sns.histplot(
        abs(model_2_pred["price"] - ensemble_pred["price"]),
        bins=50,
        alpha=0.5,
        label="CatBoost vs 融合",
    )
    sns.histplot(
        abs(model_3_pred["price"] - ensemble_pred["price"]),
        bins=50,
        alpha=0.5,
        label="LightGBM vs 融合",
    )
    plt.xlabel("与融合结果的差异")
    plt.ylabel("频数")
    plt.title("各模型与融合结果的差异分布")
    plt.legend()

    # 各模型预测值分布对比
    plt.subplot(2, 2, 4)
    sns.kdeplot(model_1_pred["price"], label="XGBoost")
    sns.kdeplot(model_2_pred["price"], label="CatBoost")
    sns.kdeplot(model_3_pred["price"], label="LightGBM")
    sns.kdeplot(ensemble_pred["price"], label="融合结果", linestyle="--")
    plt.xlabel("预测价格")
    plt.ylabel("密度")
    plt.title("各模型预测分布对比")
    plt.legend()

    plt.tight_layout()
    plt.savefig(str(Paths.Results.plots / "ensemble_analysis.png"))
    plt.close()


def save_ensemble_result(ensemble_pred):
    """
    保存融合后的预测结果
    """
    # 保存预测结果
    ensemble_pred.to_csv(
        str(Paths.Results.submission / "ensemble_3_submit_result.csv"), index=False
    )
    print("\n融合后的预测结果已保存到 ensemble_3_submit_result.csv")


def main():
    # 加载预测结果
    xgb_pred, catboost_pred, lgb_pred = load_predictions()

    # 模型融合（可以调整权重）
    ensemble_pred = ensemble_predictions(
        xgb_pred, catboost_pred, lgb_pred, weights=(0.333, 0.333, 0.334)
    )

    # 分析预测结果
    analyze_predictions(xgb_pred, catboost_pred, lgb_pred, ensemble_pred)

    # 保存结果
    save_ensemble_result(ensemble_pred)

    print("\n模型融合完成！")


if __name__ == "__main__":
    main()
