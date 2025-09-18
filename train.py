import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import optuna
import numpy as np

# 데이터 로드 및 전처리

def load_and_split_data(path, start_date='2013-01-01', val_start_date='2017-01-01', test_start_date='2018-01-01'):
    df = pd.read_csv(path, low_memory=False)
    selected_columns = [
        'loan_amnt', 'term', 'int_rate', 'grade', 'emp_length',
        'home_ownership', 'annual_inc', 'dti', 'loan_status',
        'verification_status', 'purpose', 'issue_d'
    ]
    features_df = df[selected_columns].copy().dropna()

    features_df["issue_d_datetime"] = pd.to_datetime(features_df["issue_d"], format="%b-%Y")
    features_df = features_df[features_df['issue_d_datetime'] >= pd.to_datetime(start_date)].copy()

    features_df['term'] = features_df['term'].str.replace(' months', '').astype(int)
    le = LabelEncoder()
    features_df['grade'] = le.fit_transform(features_df['grade'])
    features_df = pd.get_dummies(features_df, columns=['home_ownership'], dtype=int)
    emp_length_map = {
        '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
        '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10
    }
    features_df['emp_length'] = features_df['emp_length'].map(emp_length_map)
    features_df['loan_status'] = (features_df['loan_status'] == 'Fully Paid').astype(int)
    purpose_counts = features_df.purpose.value_counts()
    rare_purposes = purpose_counts[purpose_counts < len(features_df) * 0.1].index
    features_df['purpose'] = features_df['purpose'].replace(rare_purposes, 'others')
    features_df = pd.get_dummies(features_df, columns=['purpose', 'verification_status'], dtype=int)
    features_df["issue_month"] = features_df["issue_d_datetime"].dt.month

    # 시간순 데이터 분할
    train_df = features_df[features_df['issue_d_datetime'] < pd.to_datetime(val_start_date)]
    val_df = features_df[(features_df['issue_d_datetime'] >= pd.to_datetime(val_start_date)) & (features_df['issue_d_datetime'] < pd.to_datetime(test_start_date))]
    test_df = features_df[features_df['issue_d_datetime'] >= pd.to_datetime(test_start_date)]

    train_df.drop(columns=["issue_d", "issue_d_datetime"], inplace=True)
    val_df.drop(columns=["issue_d", "issue_d_datetime"], inplace=True)
    test_df.drop(columns=["issue_d", "issue_d_datetime"], inplace=True)

    X_train, y_train = train_df.drop("loan_status", axis=1), train_df["loan_status"]
    X_val, y_val = val_df.drop("loan_status", axis=1), val_df["loan_status"]
    X_test, y_test = test_df.drop("loan_status", axis=1), test_df["loan_status"]

    return X_train, y_train, X_val, y_val, X_test, y_test

# 공통 성능 평가 함수
def evaluate_and_log(y_true, y_proba, y_pred):
    metrics = {
        "AUC": roc_auc_score(y_true, y_proba),
        "F1": f1_score(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred)
    }
    mlflow.log_metrics(metrics)
    print("\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
    return metrics

# 로지스틱 회귀

def run_logistic(X_train, y_train, X_test, y_test, use_scaler=False):
    run_name = "Logistic_with_Scaler" if use_scaler else "Logistic"
    with mlflow.start_run(run_name=run_name):
        if use_scaler:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model = LogisticRegression(
            C=1.0, penalty='l2', solver='liblinear', max_iter=1000, class_weight='balanced', random_state=42
        )
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        mlflow.log_param("model_type", run_name)
        evaluate_and_log(y_test, y_proba, y_pred)
        mlflow.sklearn.log_model(model, "model", input_example=X_train[:5])

# LightGBM 기본 모델

def run_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test):
    with mlflow.start_run(run_name="LightGBM_basic"):
        model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            boosting_type='gbdt',
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        mlflow.log_param("model_type", "LightGBM_basic")
        evaluate_and_log(y_test, y_proba, y_pred)
        mlflow.lightgbm.log_model(model, "model", input_example=X_train[:5])

# LightGBM + Optuna

def run_lightgbm_optuna(X_train, y_train, X_val, y_val, X_test, y_test, n_trials):
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
        model = lgb.LGBMClassifier(**params, n_estimators=1000, random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        preds = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, preds)

    with mlflow.start_run(run_name="LightGBM_Optuna"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_trial.params
        model = lgb.LGBMClassifier(**best_params, n_estimators=1000, random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "LightGBM_Optuna")
        evaluate_and_log(y_test, y_proba, y_pred)
        mlflow.lightgbm.log_model(model, "model", input_example=X_train[:5])

# 메인 실행 함수
def main(args):
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(args.experiment_name)
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_data(args.data)
    run_logistic(X_train, y_train, X_test, y_test, use_scaler=False)
    run_logistic(X_train, y_train, X_test, y_test, use_scaler=True)
    run_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test)
    run_lightgbm_optuna(X_train, y_train, X_val, y_val, X_test, y_test, args.n_trials)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=r"C:/Loan_Default_Prediction_MLOps/data/accepted_2007_to_2018Q4.csv")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--experiment_name", type=str, default="Loan_Default_Evaluation")
    args = parser.parse_args()
    main(args)