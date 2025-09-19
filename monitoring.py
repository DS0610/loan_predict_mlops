import pandas as pd
import matplotlib.pyplot as plt
import os

LOG_PATH = r"C:\Loan_Default_Prediction_MLOps\data\loan_approval_2M_stratified.csv"

def monitor_predictions():
    """저장된 예측 로그를 불러와 승인(0) / 거절(1) 분포를 모니터링"""
    if not os.path.exists(LOG_PATH):
        print(f"[WARN] {LOG_PATH} 파일이 없습니다. 아직 예측 요청이 기록되지 않음.")
        return

    df = pd.read_csv(LOG_PATH)

    # 필요한 컬럼만 정제 (prediction)
    if "prediction" not in df.columns:
        print("[ERROR] 로그 파일에 'prediction' 컬럼이 없습니다.")
        return

    # 통계 
    total = len(df)
    approved = (df["prediction"] == 0).sum()   # 승인
    rejected = (df["prediction"] == 1).sum()   # 거절

    print("------ Prediction Monitoring Report ------")
    print(f"전체 요청 수: {total}")
    print(f"승인(0): {approved}건 ({approved/total:.2%})")
    print(f"거절(1): {rejected}건 ({rejected/total:.2%})")

    # 시각화 
    plt.figure(figsize=(5,4))
    df["prediction"].value_counts().sort_index().plot(
        kind="bar", color=["green", "red"]
    )
    plt.xticks([0, 1], ["승인(0)", "거절(1)"], rotation=0)
    plt.title("Prediction Distribution")
    plt.ylabel("Count")
    plt.show()

if __name__ == "__main__":
    monitor_predictions()

