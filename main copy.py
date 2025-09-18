from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
import pandas as pd
import mlflow
import mlflow.pyfunc
import joblib
import json
from contextlib import asynccontextmanager

# FastAPI Lifespan: 서버 시작 시 모델과 아티팩트를 미리 로드
# 서버의 메모리 역할을 할 딕셔너리
artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server starting up: Loading model and artifacts...")
    
    # 1. MLflow에서 최종 모델 로드
    # "LoanDefaultModel"은 train.py에서 registered_model_name으로 지정한 이름입니다.
    artifacts['model'] = mlflow.pyfunc.load_model("models:/loan_predict_LGBM/latest")
    
    # 2. 저장해 둔 전처리 아티팩트 파일 로드
    artifacts['le_grade'] = joblib.load('le_grade.joblib')
    with open('rare_purposes.json', 'r') as f:
        artifacts['rare_purposes'] = json.load(f)
    with open('training_columns.json', 'r') as f:
        artifacts['training_columns'] = json.load(f)
        
    print("Model and artifacts loaded successfully into memory.")
    yield
    # 서버 종료 시 실행 (메모리 정리)
    artifacts.clear()
    print("🔌 Server shutting down.")

# FastAPI 앱에 lifespan 관리자 등록
app = FastAPI(lifespan=lifespan)

# 예측용 데이터를 전처리하는 함수
def preprocess_for_prediction(input_data: pd.DataFrame) -> pd.DataFrame:
    df = input_data.copy()
    
    # 메모리에서 전처리 규칙 불러오기
    le_grade = artifacts['le_grade']
    rare_purposes_list = artifacts['rare_purposes']

    # 명시적 타입 변환
    # 스키마가 float(double)을 기대하는 컬럼들
    float_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'dti']
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype('float64')

    # 'term'은 문자열 처리 후 정수로 변환
    if 'term' in df.columns and isinstance(df['term'].iloc[0], str):
        df['term'] = df['term'].str.replace(' months', '').astype('int64')

    # 'grade'는 LabelEncoding 후 정수로 변환
    if 'grade' in df.columns:
        df['grade'] = df['grade'].map(lambda s: le_grade.transform([s])[0] if s in le_grade.classes_ else -1).astype('int64')

    # 'emp_length'는 매핑 후 float으로 변환 (결측 가능성 때문)
    emp_length_map = {
        '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
        '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8,
        '9 years': 9, '10+ years': 10
    }
    if 'emp_length' in df.columns:
        if isinstance(df['emp_length'].iloc[0], str):
             df['emp_length'] = df['emp_length'].map(emp_length_map)
        df['emp_length'] = df['emp_length'].astype('float64') # 최종적으로 float 변환
    
    # 'issue_d'로부터 year, month 생성 후 int32로 변환
    if 'issue_d' in df.columns:
        df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
        df["issue_year"] = df["issue_d"].dt.year.astype('int32')
        df["issue_month"] = df["issue_d"].dt.month.astype('int32')
        df = df.drop(columns=["issue_d"])

    if 'purpose' in df.columns:
        df['purpose'] = df['purpose'].replace(rare_purposes_list, 'others')

    # 원-핫 인코딩
    categorical_cols = ['home_ownership', 'purpose', 'verification_status']
    for col in categorical_cols:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], prefix=col, dtype=int)
            
    return df

# 최종 예측 엔드포인트
@app.post("/predict")
def predict(input_data: Dict[str, Any]):
    # 1. 원본 입력을 DataFrame으로 변환
    input_df = pd.DataFrame([input_data])
    
    # 2. 전처리 함수 호출
    processed_df = preprocess_for_prediction(input_df)

    # 3. 모델이 학습한 컬럼 순서와 개수에 맞춤
    training_columns = artifacts['training_columns']
    for col in training_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0 # 훈련 때 있었지만 예측 시점엔 없는 컬럼은 0으로 채움
    aligned_df = processed_df[training_columns] # 훈련 때와 동일한 순서로 정렬
    
    # 4. 예측 수행
    try:
        prediction = artifacts['model'].predict(aligned_df)
        return {"prediction_probability": float(prediction[0])}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

