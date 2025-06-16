from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

class Paths:
    class Data:
        row = project_root / "data" / "row"
        used_car_train = row / "used_car_train_20200313.csv"
        used_car_testB = row / "used_car_testB_20200421.csv"

        processed = project_root / "data" / "processed"
        x_train = processed / "x_train.joblib"
        x_val = processed / "x_val.joblib" 
        y_train = processed / "y_train.joblib"
        y_val = processed / "y_val.joblib"
        label_encoders = processed / "label_encoders.joblib"
        test_data = processed / "test_data.joblib"
        sale_ids = processed / "sale_ids.joblib"
    

    class Features:
        fe = project_root / "features"
        fe_x_train = fe / "fe_x_train.joblib"
        fe_x_val = fe / "fe_x_val.joblib"
        fe_y_train = fe / "fe_y_train.joblib"
        fe_y_val = fe / "fe_y_val.joblib"
        fe_test_data = fe / "fe_test_data.joblib"
        fe_sale_ids = fe / "fe_sale_ids.joblib"
        fe_cat_features = fe / "fe_cat_features.joblib"


    class Models:
        tmp = project_root / "models" / "tmp"


    class Results:
        plots = project_root / "results" / "plots"


