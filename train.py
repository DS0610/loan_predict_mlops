import argparse
import time
import warnings

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# 1. 데이터 관리 클래스
class DataManager:
    """데이터 로드 및 분할을 담당하는 클래스"""
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None

    def load_data(self):
        """데이터를 로드"""
        print("데이터 로딩 중...")
        try:
            self.df = pd.read_csv(self.data_path)
            print("데이터 로딩 완료. 데이터 크기:", self.df.shape)
            return self.df
        except FileNotFoundError:
            print(f"오류: '{self.data_path}' 파일을 찾을 수 없습니다.")
            print("먼저 전처리 스크립트를 실행하여 파일을 생성해주세요.")
            return None

    def split_data(self):
        """데이터를 훈련 및 테스트 세트로 분할"""
        print("학습/테스트 데이터 분할 중...")
        X = self.df.drop('loan_status', axis=1)
        y = self.df['loan_status']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("데이터 분할 완료.")
        print(f"학습 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")
        return X_train, X_test, y_train, y_test

# 2. 모델 훈련 및 평가 클래스
class ModelTrainer:
    """모델 훈련, 평가, MLflow 로깅을 담당하는 클래스"""
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

    def evaluate_and_log(self, y_true, y_pred, y_prob, model_name):
        """성능 지표를 계산하고 출력하며 MLflow에 로깅"""
        print(f"--- {model_name} 모델 성능 평가 ---")
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1_Score": f1_score(y_true, y_pred),
            "ROC_AUC": roc_auc_score(y_true, y_prob)
        }
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        
        mlflow.log_metrics(metrics)
        print("-" * 40 + "\n")

    def run_logistic_regression(self, use_scaler=False):
        """Logistic Regression 모델을 훈련하고 평가"""
        model_name = "Logistic Regression + StandardScaler" if use_scaler else "Logistic Regression (기본)"
        
        with mlflow.start_run(run_name=model_name):
            start_time = time.time()
            if use_scaler:
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('logreg', LogisticRegression(max_iter=1000, random_state=42))
                ])
            else:
                model = LogisticRegression(max_iter=1000, random_state=42)
            
            model.fit(self.X_train, self.y_train)
            
            pred = model.predict(self.X_test)
            prob = model.predict_proba(self.X_test)[:, 1]
            
            self.evaluate_and_log(self.y_test, pred, prob, model_name)
            mlflow.log_param("use_scaler", use_scaler)
            mlflow.sklearn.log_model(model, "model")
            
            elapsed_time = time.time() - start_time
            print(f"소요 시간: {elapsed_time:.2f}초")
            mlflow.log_metric("training_time", elapsed_time)

    def run_lightgbm_basic(self):
        """기본 LightGBM 모델을 훈련하고 평가"""
        model_name = "LightGBM (기본)"
        with mlflow.start_run(run_name=model_name):
            start_time = time.time()
            model = lgb.LGBMClassifier(random_state=42)
            model.fit(self.X_train, self.y_train)

            pred = model.predict(self.X_test)
            prob = model.predict_proba(self.X_test)[:, 1]

            self.evaluate_and_log(self.y_test, pred, prob, model_name)
            mlflow.lightgbm.log_model(model, "model")

            elapsed_time = time.time() - start_time
            print(f"소요 시간: {elapsed_time:.2f}초")
            mlflow.log_metric("training_time", elapsed_time)

    def run_lightgbm_optuna(self, n_trials):
        """Optuna를 사용하여 LightGBM 하이퍼파라미터를 최적화하고 평가"""
        model_name = "LightGBM + Optuna"
        
        def objective(trial):
            param = {
                'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
                'boosting_type': 'gbdt',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            }
            model = lgb.LGBMClassifier(**param)
            score = cross_val_score(model, self.X_train, self.y_train, cv=3, scoring='roc_auc').mean()
            return score

        with mlflow.start_run(run_name=model_name):
            print(f"--- {model_name} 하이퍼파라미터 튜닝 시작 (n_trials={n_trials}) ---")
            start_time = time.time()
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            
            tuning_time = time.time() - start_time
            print("튜닝 완료!")
            print(f"소요 시간: {tuning_time:.2f}초")
            print("최적 하이퍼파라미터:", study.best_params)

            mlflow.log_params(study.best_params)
            mlflow.log_metric("tuning_time", tuning_time)

            # 최적 파라미터로 최종 모델 학습 및 평가
            best_model = lgb.LGBMClassifier(**study.best_params, random_state=42)
            best_model.fit(self.X_train, self.y_train)

            pred = best_model.predict(self.X_test)
            prob = best_model.predict_proba(self.X_test)[:, 1]
            
            self.evaluate_and_log(self.y_test, pred, prob, f"{model_name} (튜닝 완료)")
            mlflow.lightgbm.log_model(best_model, "model")

# 3. 메인 실행 블록
def main(args):
    """메인 실행 함수"""
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(args.experiment_name)

    # 1. 데이터 준비
    data_manager = DataManager(args.data_path)
    if data_manager.load_data() is None:
        return
    X_train, X_test, y_train, y_test = data_manager.split_data()

    # 2. 모델 훈련 파이프라인
    trainer = ModelTrainer(X_train, X_test, y_train, y_test)
    
    trainer.run_logistic_regression(use_scaler=False)
    trainer.run_logistic_regression(use_scaler=True)
    trainer.run_lightgbm_basic()
    trainer.run_lightgbm_optuna(n_trials=args.n_trials)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loan Approval Prediction Model Training")
    parser.add_argument("--data_path", type=str, default=r"C:\Loan_Default_Prediction_MLOps\data\processed_loan_data_balanced.csv")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--experiment_name", type=str, default="Loan_Approval_Models")
    args = parser.parse_args()
    main(args)