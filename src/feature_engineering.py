# -*- coding: utf-8 -*-
"""
二手车价格预测 - 高级特征工程与CatBoost建模
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import datetime
import warnings

warnings.filterwarnings("ignore")
from config import Paths

# 设置中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_data():
    """
    加载原始数据
    """
    print("正在加载数据...")
    # 加载训练集
    train_data = pd.read_csv(str(Paths.Data.used_car_train), sep=" ")
    # 加载测试集
    test_data = pd.read_csv(str(Paths.Data.used_car_testB), sep=" ")

    print(f"训练集形状: {train_data.shape}")
    print(f"测试集形状: {test_data.shape}")

    return train_data, test_data


def preprocess_data(train_data, test_data):
    """
    数据预处理
    """
    print("\n开始数据预处理...")

    # 合并训练集和测试集进行特征工程
    train_data["source"] = "train"
    test_data["source"] = "test"
    data = pd.concat([train_data, test_data], ignore_index=True)

    # 保存SaleID
    train_ids = train_data["SaleID"]
    test_ids = test_data["SaleID"]

    # 从训练集获取y值
    y = train_data["price"]

    return data, y, train_ids, test_ids


def create_time_features(data):
    """
    创建时间特征
    """
    print("创建时间特征...")

    # 转换日期格式
    data["regDate"] = pd.to_datetime(data["regDate"], format="%Y%m%d", errors="coerce")
    data["creatDate"] = pd.to_datetime(
        data["creatDate"], format="%Y%m%d", errors="coerce"
    )

    # 处理无效日期
    data.loc[data["regDate"].isnull(), "regDate"] = pd.to_datetime(
        "20160101", format="%Y%m%d"
    )
    data.loc[data["creatDate"].isnull(), "creatDate"] = pd.to_datetime(
        "20160101", format="%Y%m%d"
    )

    # 车辆年龄（天数）
    data["vehicle_age_days"] = (data["creatDate"] - data["regDate"]).dt.days

    # 修复异常值
    data.loc[data["vehicle_age_days"] < 0, "vehicle_age_days"] = 0

    # 车辆年龄（年）
    data["vehicle_age_years"] = data["vehicle_age_days"] / 365

    # 注册年份和月份
    data["reg_year"] = data["regDate"].dt.year
    data["reg_month"] = data["regDate"].dt.month
    data["reg_day"] = data["regDate"].dt.day

    # 创建年份和月份
    data["creat_year"] = data["creatDate"].dt.year
    data["creat_month"] = data["creatDate"].dt.month
    data["creat_day"] = data["creatDate"].dt.day

    # 是否为新车（使用年限<1年）
    data["is_new_car"] = (data["vehicle_age_years"] < 1).astype(int)

    # 季节特征
    data["reg_season"] = data["reg_month"].apply(lambda x: (x % 12 + 3) // 3)
    data["creat_season"] = data["creat_month"].apply(lambda x: (x % 12 + 3) // 3)

    # 每年行驶的公里数
    data["km_per_year"] = data["kilometer"] / (data["vehicle_age_years"] + 0.1)

    # 车龄分段
    data["age_segment"] = pd.cut(
        data["vehicle_age_years"],
        bins=[0, 1, 3, 5, 10, 100],
        labels=["0-1年", "1-3年", "3-5年", "5-10年", "10年以上"],
    )
    data["age_segment"] = data["age_segment"].fillna("10年以上")

    # 构造"车龄"特征
    data["carAge"] = data["creat_year"] - data["reg_year"]

    # 可进一步提取注册月份、季度等
    data["regMonth"] = data["regDate"].dt.month
    data["regQuarter"] = data["regDate"].dt.quarter
    data["creatMonth"] = data["creatDate"].dt.month
    data["creatQuarter"] = data["creatDate"].dt.quarter

    return data


def create_car_features(data):
    """
    创建车辆特征
    """
    print("创建车辆特征...")

    # 缺失值处理
    numerical_features = [
        "power",
        "kilometer",
        "v_0",
        "v_1",
        "v_2",
        "v_3",
        "v_4",
        "v_5",
        "v_6",
        "v_7",
        "v_8",
        "v_9",
        "v_10",
        "v_11",
        "v_12",
        "v_13",
        "v_14",
    ]
    for feature in numerical_features:
        # 标记缺失值
        data[f"{feature}_missing"] = data[feature].isnull().astype(int)
        # 填充缺失值
        data[feature] = data[feature].fillna(data[feature].median())

    # 特征交互
    # 功率与排量比
    data["power_displacement_ratio"] = data["power"] / (data["v_0"] + 1)

    # 将model转换为数值型特征
    data["model_num"] = data["model"].astype("category").cat.codes

    # 品牌与车型组合
    data["brand_model"] = data["brand"].astype(str) + "_" + data["model"].astype(str)

    # 特征组合
    data["power_model"] = data["power"] + data["model_num"]

    # 相对年份特征
    current_year = datetime.datetime.now().year
    data["car_age_from_now"] = current_year - data["reg_year"]

    # 数值特征处理
    # data['power_binned'] = pd.cut(data['power'], bins=[0, 100, 200, float('inf')], labels=['0-100', '100-200', '200+'])
    data["power_log"] = data["power"].apply(lambda x: np.log(x + 1))

    # 品牌与车龄
    data["brand_carAge"] = data["brand"] * data["carAge"]

    # 车型与公里数
    data["model_kilometer"] = data["model"] * data["kilometer"]

    # 处理异常值
    numerical_cols = ["power", "kilometer", "v_0"]
    for col in numerical_cols:
        Q1 = data[col].quantile(0.05)
        Q3 = data[col].quantile(0.95)
        IQR = Q3 - Q1
        data[f"{col}_outlier"] = (
            (data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))
        ).astype(int)
        data[col] = data[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    return data


def create_statistical_features(data, train_idx):
    """
    创建统计特征
    """
    print("创建统计特征...")

    # 仅使用训练集数据创建统计特征
    train_data = data.iloc[train_idx].reset_index(drop=True)

    # 品牌级别统计
    brand_stats = (
        train_data.groupby("brand")
        .agg(
            brand_max_price=("price", "max"),
            brand_min_price=("price", "min"),
            brand_price_mean=("price", "mean"),
            brand_price_median=("price", "median"),
            brand_price_std=("price", "std"),
            brand_price_count=("price", "count"),
        )
        .reset_index()
    )

    # 车型级别统计
    model_stats = (
        train_data.groupby("model")
        .agg(
            model_price_max=("price", "max"),
            model_price_min=("price", "min"),
            model_price_mean=("price", "mean"),
            model_price_median=("price", "median"),
            model_price_std=("price", "std"),
            model_price_count=("price", "count"),
        )
        .reset_index()
    )

    # 车龄分段统计
    age_stats = (
        train_data.groupby("age_segment")
        .agg(
            age_price_max=("price", "max"),
            age_price_min=("price", "min"),
            age_price_mean=("price", "mean"),
            age_price_median=("price", "median"),
            age_price_std=("price", "std"),
        )
        .reset_index()
    )

    # 品牌 - 车型组合统计
    brand_model_stats = (
        train_data.groupby(["brand", "model"])
        .agg(
            brand_model_price_mean=("price", "mean"),
            brand_model_price_max=("price", "max"),
            brand_model_price_min=("price", "min"),
            brand_model_price_median=("price", "median"),
            brand_model_price_std=("price", "std"),
            brand_model_price_count=("price", "count"),
        )
        .reset_index()
    )

    # 合并统计特征
    data = data.merge(brand_stats, on="brand", how="left")
    data = data.merge(model_stats, on="model", how="left")
    data = data.merge(age_stats, on="age_segment", how="left")
    data = data.merge(brand_model_stats, on=["brand", "model"], how="left")

    # 相对价格特征（相对于平均价格）
    data["brand_price_ratio"] = (
        data["brand_price_mean"] / data["brand_price_mean"].mean()
    )
    data["model_price_ratio"] = (
        data["model_price_mean"] / data["model_price_mean"].mean()
    )

    # 填充缺失值
    for col in data.columns:
        if "_price_" in col and data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].median())

    return data


def encode_categorical_features(data):
    """
    编码分类特征
    """
    print("编码分类特征...")

    # 目标编码的替代方案 - 频率编码
    categorical_cols = [
        "model",
        "brand",
        "bodyType",
        "fuelType",
        "gearbox",
        "notRepairedDamage",
    ]

    for col in categorical_cols:
        # 填充缺失值
        data[col] = data[col].fillna("未知")

        # 频率编码
        freq_encoding = data.groupby(col).size() / len(data)
        data[f"{col}_freq"] = data[col].map(freq_encoding)

    # 将分类变量转换为CatBoost可以识别的格式
    for col in categorical_cols:
        data[col] = data[col].astype("str")

    return data, categorical_cols


def feature_selection(data, categorical_cols):
    """
    特征选择和最终数据准备
    """
    print("特征选择和最终数据准备...")

    # 删除不再需要的列
    drop_cols = [
        "regDate",
        "creatDate",
        "price",
        "SaleID",
        "name",
        "offerType",
        "seller",
        "source",
    ]
    data = data.drop(drop_cols, axis=1, errors="ignore")

    # 确保所有分类特征都被正确标记
    # 添加age_segment到分类特征列表中
    if "age_segment" not in categorical_cols and "age_segment" in data.columns:
        categorical_cols.append("age_segment")

    # 确保brand_model也被标记为分类特征
    if "brand_model" not in categorical_cols and "brand_model" in data.columns:
        categorical_cols.append("brand_model")

    # 转换分类特征
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype("category")

    return data, categorical_cols


def main():
    # 加载数据
    train_data, test_data = load_data()

    # 预处理数据
    data, y, train_ids, test_ids = preprocess_data(train_data, test_data)

    # 创建时间特征
    data = create_time_features(data)

    # 创建车辆特征
    data = create_car_features(data)

    # 找回训练集的索引
    train_idx = data[data["source"] == "train"].index
    test_idx = data[data["source"] == "test"].index

    # 创建统计特征
    data = create_statistical_features(data, train_idx)

    # 编码分类特征
    data, categorical_cols = encode_categorical_features(data)

    # 特征选择和最终准备
    data, cat_features = feature_selection(data, categorical_cols)

    # 分离训练集和测试集
    x_train_full = data.iloc[train_idx].reset_index(drop=True)
    x_test = data.iloc[test_idx].reset_index(drop=True)

    # 划分训练集和验证集
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y, test_size=0.2, random_state=42
    )

    # 保存处理后的数据
    joblib.dump(x_train, str(Paths.Features.fe_x_train))
    joblib.dump(x_val, str(Paths.Features.fe_x_val))
    joblib.dump(y_train, str(Paths.Features.fe_y_train))
    joblib.dump(y_val, str(Paths.Features.fe_y_val))
    joblib.dump(x_test, str(Paths.Features.fe_test_data))
    joblib.dump(test_ids, str(Paths.Features.fe_sale_ids))
    joblib.dump(cat_features, str(Paths.Features.fe_cat_features))

    print("预处理后的数据已保存")


if __name__ == "__main__":
    main()
