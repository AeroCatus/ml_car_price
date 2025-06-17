# -*- coding: utf-8 -*-
"""
二手车价格预测 - XGBoost建模（直接复用特征工程结果）
"""

from config import Paths
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载特征工程后的数据
x_train = joblib.load(str(Paths.Features.fe_x_train))
x_val = joblib.load(str(Paths.Features.fe_x_val))
y_train = joblib.load(str(Paths.Features.fe_y_train))
y_val = joblib.load(str(Paths.Features.fe_y_val))
x_test = joblib.load(str(Paths.Features.fe_test_data))
test_ids = joblib.load(str(Paths.Features.fe_sale_ids))

# 2. XGBoost训练
print("开始训练XGBoost模型...")

# XGBoost不支持直接用category类型，需转为int
for col in x_train.select_dtypes(include='category').columns:
    x_train[col] = x_train[col].cat.codes
    x_val[col] = x_val[col].cat.codes
    x_test[col] = x_test[col].cat.codes

dtrain = xgb.DMatrix(x_train, label=y_train)
dval = xgb.DMatrix(x_val, label=y_val)
dtest = xgb.DMatrix(x_test)

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'learning_rate': 0.01,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42,
    'nthread': -1
}

evals = [(dtrain, 'train'), (dval, 'val')]
model = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    evals=evals,
    early_stopping_rounds=20,
    verbose_eval=100
)

joblib.dump(model, str(Paths.Models.xgboost / 'fe_xgb_model.joblib'))

# 3. 验证集评估
y_pred_val = model.predict(dval)
mse = mean_squared_error(y_val, y_pred_val)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_val, y_pred_val)
r2 = r2_score(y_val, y_pred_val)

print("\n模型评估结果：")
print(f"均方根误差 (RMSE): {rmse:.2f}")
print(f"平均绝对误差 (MAE): {mae:.2f}")
print(f"R2分数: {r2:.4f}")

# 4. 特征重要性
importance = model.get_score(importance_type='weight')
importance_df = pd.DataFrame({
    'feature': list(importance.keys()),
    'importance': list(importance.values())
}).sort_values('importance', ascending=False)
importance_df.to_csv(str(Paths.Results.importance / 'fe_xgb_feature_importance.csv'), index=False)

plt.figure(figsize=(14, 8))
sns.barplot(x='importance', y='feature', data=importance_df.head(20))
plt.title('XGBoost Top 20 特征重要性')
plt.tight_layout()
plt.savefig(str(Paths.Results.importance / 'fe_xgb_feature_importance.png'))
plt.close()

# 5. 预测测试集并保存
y_pred_test = model.predict(dtest)
submit_data = pd.DataFrame({
    'SaleID': test_ids,
    'price': y_pred_test
})
submit_data.to_csv(str(Paths.Models.xgboost / 'fe_xgb_submit_result.csv'), index=False)
print("预测结果已保存到 fe_xgb_submit_result.csv")