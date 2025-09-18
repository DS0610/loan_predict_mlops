import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import json

def create_and_save_artifacts(path: str):
    """
    훈련 데이터셋에서 전처리 규칙(아티팩트)을 학습하고 파일로 저장합니다.
    - LabelEncoder 객체
    - Rare purposes 리스트
    - 최종 훈련 컬럼 리스트
    """
    print(f"Loading data from {path}...")
    df = pd.read_csv(path, low_memory=False)
    selected_columns = [
        'loan_amnt', 'term', 'int_rate', 'grade', 'emp_length',
        'home_ownership', 'annual_inc', 'dti', 'loan_status',
        'verification_status', 'purpose', 'issue_d'
    ]
    features_df = df[selected_columns].copy().dropna()

    # 1. LabelEncoder 학습 및 저장 
    le_grade = LabelEncoder()
    le_grade.fit(features_df['grade']) # 'grade' 컬럼의 모든 고유값을 학습
    joblib.dump(le_grade, 'le_grade.joblib')
    print("Saved 'le_grade.joblib'")
    # 학습된 인코더로 데이터 변환
    features_df['grade'] = le_grade.transform(features_df['grade'])

    # 2. Rare Purposes 리스트 생성 및 저장
    purpose_counts = features_df.purpose.value_counts()
    threshold = len(features_df) * 0.1
    rare_purposes = purpose_counts[purpose_counts < threshold].index
    rare_purposes_list = list(rare_purposes)
    with open('rare_purposes.json', 'w') as f:
        json.dump(rare_purposes_list, f)
    print("Saved 'rare_purposes.json'")
    # 리스트를 사용해 데이터 변환
    features_df['purpose'] = features_df['purpose'].replace(rare_purposes, 'others')

    # 3. 최종 훈련 컬럼 리스트를 얻기 위한 나머지 전처리 과정 
    # 이 부분은 train.py와 동일하게 수행하여 최종 컬럼 형태를 만듭니다.
    features_df['term'] = features_df['term'].str.replace(' months', '').astype(int)
    features_df = pd.get_dummies(features_df, columns=['home_ownership'], prefix='home_ownership', dtype=int)
    emp_length_map = {
        '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
        '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8,
        '9 years': 9, '10+ years': 10
    }
    features_df['emp_length'] = features_df['emp_length'].map(emp_length_map)
    features_df['loan_status'] = (features_df['loan_status'] == 'Fully Paid').astype(int)
    features_df = pd.get_dummies(features_df, columns=['purpose'], prefix='purpose', dtype=int)
    features_df = pd.get_dummies(features_df, columns=['verification_status'], prefix='verification_status', dtype=int)
    features_df["issue_d"] = pd.to_datetime(features_df["issue_d"], format="%b-%Y")
    features_df["issue_year"] = features_df["issue_d"].dt.year
    features_df["issue_month"] = features_df["issue_d"].dt.month
    features_df = features_df.drop(columns=["issue_d"])

    X = features_df.drop("loan_status", axis=1)
    
    # 4. 최종 컬럼 리스트 저장 
    training_columns = X.columns.tolist()
    with open('training_columns.json', 'w') as f:
        json.dump(training_columns, f)
    print("Saved 'training_columns.json'")
    print("\n All artifacts have been created successfully!")


if __name__ == "__main__":
    # train.py에서 사용했던 것과 동일한 훈련 데이터 파일 경로를 지정해야 합니다.
    # argparse를 사용하지 않고 직접 경로를 지정합니다.
    TRAINING_DATA_PATH = r"C:\Loan_Default_Prediction_MLOps\data\accepted_2007_to_2018Q4.csv"
    create_and_save_artifacts(TRAINING_DATA_PATH)


# ```

# ### **사용 방법**

# 1.  **코드 저장:** 위 코드를 `save_artifacts.py`라는 이름으로 `train.py`가 있는 프로젝트 폴더에 저장합니다.
# 2.  **경로 확인:** 코드 맨 아래 `TRAINING_DATA_PATH` 변수의 값이 `train.py`에서 사용했던 데이터 경로와 일치하는지 다시 한번 확인합니다.
# 3.  **스크립트 실행:** 터미널을 열고 프로젝트 폴더로 이동한 뒤, 아래 명령어를 입력하여 스크립트를 **딱 한 번만 실행**합니다.

#     ```bash
#     python save_artifacts.py
    

