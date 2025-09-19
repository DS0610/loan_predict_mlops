from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.lightgbm
from save_artifacts import model
from datetime import datetime
import os

app = FastAPI(title="Loan Default Prediction API")

# 설정 
THRESHOLD = 0.4  # 예측 기준값
LOG_PATH = "predictions_log.csv"

# 입력 스키마 (간단 입력) 
class LoanApplication(BaseModel):
    amount_requested: float
    employment_length: int
    dti: float
    state: str   # 예: "CA", "TX", "NV"
    zip_prefix: int = 0   # 기본값 0


@app.get("/")
def root():
    return {"message": "Loan Default Prediction API is running"}


@app.post("/predict")
def predict(app_data: LoanApplication):
    try:
        # 모델 학습 피처 목록 불러오기
        expected_features = model.booster_.feature_name()
        input_dict = {f: 0 for f in expected_features}  # 전부 0으로 초기화

        # 필수 feature 채우기
        input_dict["amount_requested"] = app_data.amount_requested
        input_dict["employment_length"] = app_data.employment_length
        input_dict["dti"] = app_data.dti
        input_dict["zip_prefix"] = app_data.zip_prefix

        # 날짜 자동 생성
        now = datetime.now()
        input_dict["issue_year"] = now.year
        input_dict["issue_month"] = now.month

        # state 원핫 인코딩
        state_col = f"state_{app_data.state.upper()}"
        if state_col in input_dict:
            input_dict[state_col] = 1
        else:
            raise HTTPException(status_code=400, detail=f"지원하지 않는 state 입력: {app_data.state}")

        # DataFrame 변환
        input_df = pd.DataFrame([input_dict])[expected_features]

        # 예측
        proba = model.predict_proba(input_df)[:, 1][0]
        raw_pred = int(proba >= THRESHOLD)

        prediction = 1 - raw_pred

        result = {
            "prediction": prediction,
            "probability": float(proba),
            "used_state": state_col,
            "issue_year": input_dict["issue_year"],
            "issue_month": input_dict["issue_month"],
            "threshold": THRESHOLD
        }

        # 로그 저장
        log_df = pd.DataFrame([{
            **app_data.dict(),
            "prediction": prediction,
            "probability": float(proba),
            "issue_year": input_dict["issue_year"],
            "issue_month": input_dict["issue_month"],
            "timestamp": now
        }])
        log_df.to_csv(LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
