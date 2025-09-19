import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import optuna
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import warnings

warnings.filterwarnings('ignore')

# 1. 데이터 관리 클래스
class DataManager:
    """데이터 로드, 전처리, 분할을 담당하는 클래스"""

    def __init__(self, data_path):
        """
        Args:
            data_path (str): 데이터 파일 경로
        """
        self.data_path = data_path
        self.features_df = None

    def load_data(self):
        """데이터를 로드하고 초기 피처를 선택"""
        print("데이터 로딩 중...")
        df = pd.read_csv(self.data_path, low_memory=False)

        # 데이터 누수 가능성이 높은 'risk_score', 'policy_code'를 제외하고 피처 선택
        selected_columns = [
            'amount_requested', 'employment_length', 'application_date',
            'loan_title', 'zip_code', 'state', 'dti', 'target'
        ]
        self.features_df = df[selected_columns].copy()
        print("데이터 로딩 완료.")

    def preprocess(self):
        """데이터 전처리를 수행"""
        print("데이터 전처리 중...")
        df = self.features_df

        # 날짜 피처 생성
        df["application_date"] = pd.to_datetime(df["application_date"], errors="coerce")
        df["issue_year"] = df["application_date"].dt.year.astype("Int16")
        df["issue_month"] = df["application_date"].dt.month.astype("Int8")

        # employment_length 숫자형으로 변환
        emp_length_map = {
            '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
            '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10
        }
        df['employment_length'] = df['employment_length'].map(emp_length_map)

        # dti 피처 정리
        df["dti"] = pd.to_numeric(df["dti"].astype(str).str.replace("%", "", regex=False), errors="coerce").astype("float32")

        # zip_code 전처리 (앞 3자리만 사용) 및 라벨 인코딩
        df["zip_prefix"] = df["zip_code"].astype(str).str[:3]
        le = LabelEncoder()
        df["zip_prefix"] = le.fit_transform(df["zip_prefix"].astype(str))

        # state 원핫 인코딩
        df = pd.get_dummies(df, columns=["state"], prefix="state", dtype=int)

        # 불필요한 원본 컬럼 삭제
        df.drop(columns=["application_date", "zip_code", "loan_title"], inplace=True, errors="ignore")

        self.features_df = df
        print("데이터 전처리 완료.")

    def split_data(self, val_year=2017, test_year=2018):
        """시간 기준으로 데이터를 훈련, 검증, 테스트 세트로 분할"""
        print("데이터 분할 중...")
        train_df = self.features_df[self.features_df["issue_year"] < val_year]
        val_df = self.features_df[(self.features_df["issue_year"] >= val_year) & (self.features_df["issue_year"] < test_year)]
        test_df = self.features_df[self.features_df["issue_year"] >= test_year]

        X_train, y_train = train_df.drop("target", axis=1), train_df["target"]
        X_val, y_val = val_df.drop("target", axis=1), val_df["target"]
        X_test, y_test = test_df.drop("target", axis=1), test_df["target"]
        
        # NaN 값 처리
        imputer = SimpleImputer(strategy="median")
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
        X_val = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
        X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
        
        print("데이터 분할 완료.")
        return X_train, y_train, X_val, y_val, X_test, y_test

# 2. 모델 훈련 및 평가 클래스

class ModelTrainer:
    """모델 훈련, 평가, MLflow 로깅을 담당하는 클래스"""

    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.data = {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test
        }

    def evaluate_and_log(self, y_true, y_proba, y_pred, model_name):
        """성능 지표를 계산하고 MLflow에 로깅"""
        print(f"--- {model_name} 평가 결과 ---")
        metrics = {
            "AUC": roc_auc_score(y_true, y_proba),
            "F1": f1_score(y_true, y_pred),
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred)
        }
        mlflow.log_metrics(metrics)
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        return metrics

    def run_logistic(self, use_scaler=False):
        """Logistic Regression 모델을 훈련하고 평가"""
        run_name = "Logistic_with_Scaler" if use_scaler else "Logistic"
        with mlflow.start_run(run_name=run_name):
            X_train, y_train = self.data["X_train"], self.data["y_train"]
            X_test, y_test = self.data["X_test"], self.data["y_test"]

            if use_scaler:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
            model = LogisticRegression(
                C=1.0, penalty="l2", solver="liblinear",
                max_iter=1000, class_weight="balanced", random_state=42
            )
            model.fit(X_train, y_train)

            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            mlflow.log_param("model_type", run_name)
            self.evaluate_and_log(y_test, y_proba, y_pred, run_name)
            mlflow.sklearn.log_model(model, "model", input_example=pd.DataFrame(X_train[:5], columns=self.data["X_train"].columns))

    def run_lightgbm(self):
        """기본 LightGBM 모델을 훈련하고 평가"""
        run_name = "LightGBM_basic"
        with mlflow.start_run(run_name=run_name):
            d = self.data
            model = lgb.LGBMClassifier(
                objective="binary", metric="auc",
                n_estimators=500, learning_rate=0.05,
                max_depth=6, num_leaves=31,
                class_weight="balanced", random_state=42
            )
            model.fit(d["X_train"], d["y_train"], 
                      eval_set=[(d["X_val"], d["y_val"])], 
                      callbacks=[lgb.early_stopping(50, verbose=False)])
            
            y_proba = model.predict_proba(d["X_test"])[:, 1]
            y_pred = model.predict(d["X_test"])

            mlflow.log_param("model_type", run_name)
            self.evaluate_and_log(d["y_test"], y_proba, y_pred, run_name)
            mlflow.lightgbm.log_model(model, "model", input_example=d["X_train"][:5])

    def run_lightgbm_optuna(self, n_trials):
        """Optuna를 사용하여 LightGBM 하이퍼파라미터를 최적화하고 평가"""
        d = self.data
        
        # SMOTE는 훈련 데이터에만 적용
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(d["X_train"], d["y_train"])

        def objective(trial):
            params = {
                "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
            model = lgb.LGBMClassifier(**params, n_estimators=1000, random_state=42)
            model.fit(X_train_smote, y_train_smote, eval_set=[(d["X_val"], d["y_val"])], callbacks=[lgb.early_stopping(50, verbose=False)])
            preds = model.predict_proba(d["X_val"])[:, 1]
            return roc_auc_score(d["y_val"], preds)

        run_name = "LightGBM_Optuna"
        with mlflow.start_run(run_name=run_name):
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)
            best_params = study.best_trial.params

            model = lgb.LGBMClassifier(**best_params, n_estimators=1000, random_state=42)
            model.fit(X_train_smote, y_train_smote, eval_set=[(d["X_val"], d["y_val"])], callbacks=[lgb.early_stopping(50, verbose=False)])
            
            y_proba = model.predict_proba(d["X_test"])[:, 1]
            y_pred = model.predict(d["X_test"])

            mlflow.log_params(best_params)
            mlflow.log_param("model_type", run_name)
            self.evaluate_and_log(d["y_test"], y_proba, y_pred, run_name)
            mlflow.lightgbm.log_model(model, "model", input_example=d["X_train"][:5])


# 3. 메인 실행 블록
def main(args):
    """메인 실행 함수"""
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(args.experiment_name)

    # 1. 데이터 준비
    data_manager = DataManager(args.data)
    data_manager.load_data()
    data_manager.preprocess()
    X_train, y_train, X_val, y_val, X_test, y_test = data_manager.split_data()

    # 2. 모델 훈련 및 평가
    trainer = ModelTrainer(X_train, y_train, X_val, y_val, X_test, y_test)
    
    print("\n[실험 1] Logistic Regression (No Scaler)")
    trainer.run_logistic(use_scaler=False)
    
    print("\n[실험 2] Logistic Regression (With Scaler)")
    trainer.run_logistic(use_scaler=True)

    print("\n[실험 3] LightGBM (Basic)")
    trainer.run_lightgbm()
    
    print(f"\n[실험 4] LightGBM with Optuna (Trials: {args.n_trials})")
    trainer.run_lightgbm_optuna(args.n_trials)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loan Default Prediction Model Training")
    parser.add_argument("--data", type=str, default=r"C:\Loan_Default_Prediction_MLOps\data\loan_approval_2M_stratified.csv")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--experiment_name", type=str, default="Loan_Approval_Refactored")
    args = parser.parse_args()
    main(args)
