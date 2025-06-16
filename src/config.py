from pathlib import Path

project_root = Path(__file__).resolve().parent.parent


class Paths:
    class Data:
        data = project_root / "data"
        data.mkdir(parents=True, exist_ok=True)
        used_car_train = data / "used_car_train_20200313.csv"
        used_car_testB = data / "used_car_testB_20200421.csv"

    class Features:
        fe = project_root / "features"
        fe.mkdir(parents=True, exist_ok=True)
        fe_x_train = fe / "fe_x_train.joblib"
        fe_x_val = fe / "fe_x_val.joblib"
        fe_y_train = fe / "fe_y_train.joblib"
        fe_y_val = fe / "fe_y_val.joblib"
        fe_test_data = fe / "fe_test_data.joblib"
        fe_sale_ids = fe / "fe_sale_ids.joblib"
        fe_cat_features = fe / "fe_cat_features.joblib"

    class Models:
        catboost = project_root / "models" / "catboost"
        catboost.mkdir(parents=True, exist_ok=True)
        ligjhtgbm = project_root / "models" / "lightgbm"
        ligjhtgbm.mkdir(parents=True, exist_ok=True)
        xgboost = project_root / "models" / "xgboost"
        xgboost.mkdir(parents=True, exist_ok=True)

    class Results:
        plots = project_root / "results" / "plots"
        plots.mkdir(parents=True, exist_ok=True)
        importance = project_root / "results" / "importance"
        importance.mkdir(parents=True, exist_ok=True)
        submission = project_root / "results" / "submission"
        submission.mkdir(parents=True, exist_ok=True)
