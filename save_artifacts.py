import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import os

# 전처리 규칙(아티팩트) 생성 및 저장
def create_and_save_artifacts(path: str, start_date='2013-01-01'):
    print(f"Loading data from {path} to create artifacts...")
    df = pd.read_csv(path, low_memory=False)
    selected_columns = [
        'loan_amnt', 'term', 'int_rate', 'grade', 'emp_length',
        'home_ownership', 'annual_inc', 'dti', 'loan_status',
        'verification_status', 'purpose', 'issue_d'
    ]
    features_df = df[selected_columns].copy().dropna()
    
    features_df["issue_d_datetime"] = pd.to_datetime(features_df["issue_d"], format="%b-%Y")
    features_df = features_df[features_df['issue_d_datetime'] >= pd.to_datetime(start_date)].copy()

    # 1. LabelEncoder 학습 및 저장
    le_grade = LabelEncoder()
    le_grade.fit(features_df['grade'])
    joblib.dump(le_grade, 'le_grade.joblib')
    print("Artifact 'le_grade.joblib' saved.")
    features_df['grade'] = le_grade.transform(features_df['grade'])

    # 2. Rare Purposes 리스트 생성 및 저장
    purpose_counts = features_df.purpose.value_counts()
    threshold = len(features_df) * 0.1
    rare_purposes = purpose_counts[purpose_counts < threshold].index
    with open('rare_purposes.json', 'w') as f:
        json.dump(list(rare_purposes), f)
    print("Artifact 'rare_purposes.json' saved.")
    features_df['purpose'] = features_df['purpose'].replace(rare_purposes, 'others')

    # 3. 최종 훈련 컬럼 리스트 생성을 위한 나머지 전처리
    features_df['term'] = features_df['term'].str.replace(' months', '').astype(int)
    features_df = pd.get_dummies(features_df, columns=['home_ownership'], prefix='home_ownership', dtype=int)
    emp_length_map = {
        '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
        '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10
    }
    features_df['emp_length'] = features_df['emp_length'].map(emp_length_map)
    features_df['loan_status'] = (features_df['loan_status'] == 'Fully Paid').astype(int)
    features_df = pd.get_dummies(features_df, columns=['purpose'], prefix='purpose', dtype=int)
    features_df = pd.get_dummies(features_df, columns=['verification_status'], prefix='verification_status', dtype=int)
    features_df["issue_month"] = features_df["issue_d_datetime"].dt.month
    
    features_df = features_df.drop(columns=["issue_d", "issue_d_datetime"])
    
    X = features_df.drop("loan_status", axis=1)
    
    # 4. 최종 컬럼 리스트 저장
    with open('training_columns.json', 'w') as f:
        json.dump(X.columns.tolist(), f)
    print("Artifact 'training_columns.json' saved.")
    print("\nAll artifacts created successfully.")

if __name__ == "__main__":
    # 이 경로는 train.py의 default 경로와 동일하게 설정합니다.
    data_path = r"C:/Loan_Default_Prediction_MLOps/data/accepted_2007_to_2018Q4.csv"
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
    else:
        create_and_save_artifacts(data_path)

