from mlflow.tracking import MlflowClient

client = MlflowClient()
client.set_registered_model_alias(
    name="final_loan_LGBM_OPTUNA",
    alias="production",
    version="1"
)

print("[INFO] v1 -> alias=production 등록 완료")