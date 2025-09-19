import mlflow
import mlflow.lightgbm

# 모델 불러오기 함수
def load_model(model_name="final_loan_LGBM_OPTUNA", model_version=1):
    mlflow.set_tracking_uri("file:./mlruns")
    model_uri = f"models:/{model_name}/{model_version}"
    print(f"[INFO] Loading model from {model_uri}")
    model = mlflow.lightgbm.load_model(model_uri)
    return model

# 모듈 import 시점에서 미리 로드해두기
model = load_model()

if __name__ == "__main__":
    features = model.booster_.feature_name()
    print("총 feature 개수:", len(features))
    for f in features:
        print(f)


