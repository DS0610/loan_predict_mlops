import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score
)
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
import json


# 데이터 로드 및 전처리
def load_and_preprocess(path):
    df = pd.read_csv(path, low_memory=False)
    selected_columns = [
        'loan_amnt', 'term', 'int_rate', 'grade', 'emp_length',
        'home_ownership', 'annual_inc', 'dti', 'loan_status',
        'verification_status', 'purpose', 'issue_d'
    ]
    features_df = df[selected_columns].copy().dropna()

    features_df['term'] = features_df['term'].str.replace(' months', '').astype(int)

    le = LabelEncoder()
    features_df['grade'] = le.fit_transform(features_df['grade'])

    features_df = pd.get_dummies(features_df, columns=['home_ownership'], prefix='home_ownership', dtype=int)

    emp_length_map = {
        '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
        '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8,
        '9 years': 9, '10+ years': 10
    }
    features_df['emp_length'] = features_df['emp_length'].map(emp_length_map)

    features_df['loan_status'] = (features_df['loan_status'] == 'Fully Paid').astype(int)

    purpose1 = features_df.purpose.value_counts()
    threshold = len(features_df) * 0.1
    rare_purposes = purpose1[purpose1 < threshold].index
    features_df['purpose_cleaned'] = features_df['purpose'].replace(rare_purposes, 'others')
    features_df = pd.get_dummies(features_df, columns=['purpose_cleaned'], prefix='purpose', dtype=int)
    features_df = features_df.drop('purpose', axis=1)

    features_df = pd.get_dummies(features_df, columns=['verification_status'], prefix='verification_status', dtype=int)

    features_df["issue_d"] = pd.to_datetime(features_df["issue_d"], format="%b-%Y")
    features_df["issue_year"] = features_df["issue_d"].dt.year
    features_df["issue_month"] = features_df["issue_d"].dt.month
    features_df = features_df.drop(columns=["issue_d"])

    X = features_df.drop("loan_status", axis=1)
    y = features_df["loan_status"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


# 공통 평가 함수
def evaluate_and_log(y_true, y_pred, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    mlflow.log_metric("AUC", auc)
    mlflow.log_metric("F1", f1)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("Accuracy", accuracy)

    return {"auc": auc, "f1": f1, "precision": precision, "recall": recall, "accuracy": accuracy}


# Logistic Regression 
def run_logistic(X_train, X_test, y_train, y_test, use_scaler=False):
    run_name = "LogisticRegression+Scaler" if use_scaler else "LogisticRegression"
    with mlflow.start_run(run_name=run_name):
        if use_scaler:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model = LogisticRegression(max_iter=5000, solver="lbfgs")
        model.fit(X_train, y_train)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("max_iter", 5000)
        mlflow.log_param("solver", "lbfgs")

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        evaluate_and_log(y_test, y_pred, y_prob)

        # registered_model_name 인자 제거
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=X_train[:5]
        )


# LightGBM 
def run_lightgbm(X_train, X_test, y_train, y_test, params=None):
    with mlflow.start_run(run_name="LightGBM"):
        if params is None:
            params = {"objective": "binary", "metric": "auc", "verbosity": -1}

        model = lgb.LGBMClassifier(**params, n_estimators=500, random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="auc")

        mlflow.log_param("model_type", "LightGBM")
        mlflow.log_params(params)
        mlflow.log_param("n_estimators", 500)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        evaluate_and_log(y_test, y_pred, y_prob)

        fig, ax = plt.subplots(figsize=(8, 6))
        lgb.plot_importance(model, ax=ax)
        plt.title("LightGBM Feature Importance")
        plt.tight_layout()
        mlflow.log_figure(fig, "feature_importance.png")

        # registered_model_name 인자 제거
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            input_example=X_train[:5]
        )


# LightGBM + Optuna 
def run_lightgbm_optuna(X_train, X_test, y_train, y_test, n_trials=20):
    def objective(trial, X_train, X_valid, y_train, y_valid):
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 512),
            "max_depth": trial.suggest_int("max_depth", 3, 16),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0)
        }
        model = lgb.LGBMClassifier(**params, n_estimators=500, random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric="auc")
        preds = model.predict_proba(X_valid)[:, 1]
        return roc_auc_score(y_valid, preds)

    with mlflow.start_run(run_name="LightGBM+Optuna"):
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_tr, X_val, y_tr, y_val), n_trials=n_trials)

        best_params = study.best_trial.params
        mlflow.log_param("model_type", "LightGBM+Optuna")
        mlflow.log_params(best_params)
        mlflow.log_param("n_trials", n_trials)

        with open("best_params.json", "w") as f:
            json.dump(best_params, f)
        mlflow.log_artifact("best_params.json")

        model = lgb.LGBMClassifier(**best_params, n_estimators=500, random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="auc")

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        evaluate_and_log(y_test, y_pred, y_prob)

        # registered_model_name 인자 제거
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            input_example=X_train[:5]
        )


# Main
def main(args):
    mlflow.set_tracking_uri("file:C:/Loan_Default_Prediction_MLOps/mlruns")
    mlflow.set_experiment("loan_default_predict_model_experiment_re1")

    X_train, X_test, y_train, y_test = load_and_preprocess(args.data)

    run_logistic(X_train, X_test, y_train, y_test, use_scaler=False)
    run_logistic(X_train, X_test, y_train, y_test, use_scaler=True)
    run_lightgbm(X_train, X_test, y_train, y_test)
    run_lightgbm_optuna(X_train, X_test, y_train, y_test, n_trials=args.n_trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="C:/Loan_Default_Prediction_MLOps/data/accepted_2007_to_2018Q4.csv")
    parser.add_argument("--n_trials", type=int, default=20)
    args = parser.parse_args()
    main(args)
