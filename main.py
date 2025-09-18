from fastapi import FastAPI
from typing import Dict, Any
import uvicorn
import pandas as pd
import mlflow
import mlflow.pyfunc
import joblib
import json
from contextlib import asynccontextmanager

# 서버 메모리 역할을 할 딕셔너리
artifacts = {}

# 서버 시작 시 모델과 아티팩트를 미리 로드
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model and artifacts...")
    # MLflow UI에서 수동 등록한 모델 이름을 사용합니다.
    artifacts['model'] = mlflow.pyfunc.load_model("models:/LoanDefaultModel/latest")
    
    # save_artifacts.py로 생성된 파일들을 로드합니다.
    artifacts['le_grade'] = joblib.load('le_grade.joblib')
    with open('rare_purposes.json', 'r') as f:
        artifacts['rare_purposes'] = json.load(f)
    with open('training_columns.json', 'r') as f:
        artifacts['training_columns'] = json.load(f)
        
    print("Model and artifacts loaded successfully.")
    yield
    # 서버 종료 시 메모리 정리
    artifacts.clear()

app = FastAPI(lifespan=lifespan)

# 예측용 데이터 전처리
def preprocess_for_prediction(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()
    
    # 메모리에서 전처리 규칙 불러오기
    le_grade = artifacts['le_grade']
    rare_purposes_list = artifacts['rare_purposes']

    # 원본 데이터 타입에 맞게 전처리
    if 'term' in df.columns:
        df['term'] = df['term'].str.replace(' months', '').astype(int)
    if 'grade' in df.columns:
        # 학습된 인코더의 클래스에 없는 새로운 값이 들어올 경우를 대비
        df['grade'] = df['grade'].map(lambda s: le_grade.transform([s])[0] if s in le_grade.classes_ else -1)
    
    emp_length_map = {
        '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
        '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10
    }
    if 'emp_length' in df.columns:
        df['emp_length'] = df['emp_length'].map(emp_length_map)
        
    if 'issue_d' in df.columns:
        df["issue_d_datetime"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
        df["issue_month"] = df["issue_d_datetime"].dt.month
        df = df.drop(columns=["issue_d", "issue_d_datetime"])
        
    if 'purpose' in df.columns:
        df['purpose'] = df['purpose'].replace(rare_purposes_list, 'others')
    
    # 원-핫 인코딩
    categorical_cols = ['home_ownership', 'purpose', 'verification_status']
    for col in categorical_cols:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], prefix=col, dtype=int)
            
    return df

# 예측 엔드포인트
@app.post("/predict")
def predict(input_data: Dict[str, Any]):
    input_df = pd.DataFrame([input_data])
    processed_df = preprocess_for_prediction(input_df)
    
    # 훈련 시점의 컬럼 순서와 개수에 맞춤
    training_columns = artifacts['training_columns']
    for col in training_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0
    aligned_df = processed_df[training_columns]
    
    try:
        # 모델 스키마에 따라 최종 타입 변환
        model_schema = artifacts['model'].metadata.get_input_schema()
        type_map = {'integer': 'int32', 'long': 'int64', 'float': 'float32', 'double': 'float64'}
        cast_dict = {}
        if model_schema:
            for col_info in model_schema.to_dict()['inputs']:
                col_name, col_type = col_info['name'], col_info['type']
                if col_name in aligned_df.columns:
                    target_type = type_map.get(col_type)
                    if target_type and aligned_df[col_name].dtype.name != target_type:
                        cast_dict[col_name] = target_type
        if cast_dict:
            aligned_df = aligned_df.astype(cast_dict)

        prediction = artifacts['model'].predict(aligned_df)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)