import mlflow
import mlflow.lightgbm
import matplotlib.pyplot as plt
import pandas as pd

MODEL_NAME = "final_loan_LGBM_OPTUNA"
MODEL_VERSION = 1

def load_model(model_name=MODEL_NAME, model_version=MODEL_VERSION):
    mlflow.set_tracking_uri("file:./mlruns")
    model_uri = f"models:/{model_name}/{model_version}"
    print(f"[INFO] Loading model from {model_uri}")
    model = mlflow.lightgbm.load_model(model_uri)
    return model

if __name__ == "__main__":
    model = load_model()

    # booster에서 feature importance 추출
    booster = model.booster_
    feature_names = booster.feature_name()
    importances = booster.feature_importance(importance_type="gain")

    # DataFrame 변환
    df_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    print(df_importance.head(20))  # 상위 20개 출력

    # 시각화
    plt.figure(figsize=(10, 8))
    plt.barh(df_importance["feature"][:20][::-1], df_importance["importance"][:20][::-1])
    plt.xlabel("Feature Importance (gain)")
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.show()
