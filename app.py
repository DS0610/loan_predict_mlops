import streamlit as st
import pandas as pd
import mlflow
import mlflow.lightgbm
from datetime import datetime
import os

# 설정
MODEL_NAME = "final_loan_LGBM_OPTUNA"
MODEL_VERSION = 1
LOG_PATH = "predictions_log.csv"

# 모델 불러오기
@st.cache_resource
def load_model(model_name=MODEL_NAME, model_version=MODEL_VERSION):
    mlflow.set_tracking_uri("file:./mlruns")
    model_uri = f"models:/{model_name}/{model_version}"
    st.info(f"[INFO] Loading model from {model_uri}")
    model = mlflow.lightgbm.load_model(model_uri)
    return model

model = load_model()


# 사이드바 메뉴
menu = st.sidebar.radio("메뉴 선택", ["📝 실시간 예측", "📊 모니터링"])

# 실시간 예측
if menu == "📝 실시간 예측":
    st.title("📝 실시간 대출 예측")
    st.markdown("대출 신청 정보를 입력하면 승인(0) / 거절(1) 여부를 예측합니다.")

    with st.form("loan_form"):
        amount_requested = st.slider(   "대출 신청 금액 (USD)", min_value=500, max_value=50000, step=500, value=5000)
        employment_length = st.slider("근속 연수", 0, 20, 5)
        dti = st.slider("DTI (부채/소득 비율)", min_value=0.0, max_value=100.0, step=0.1, value=10.0)
        state = st.selectbox("거주 주(State)", ["CA", "TX", "FL", "NY", "NV"])
        zip_prefix = st.text_input("우편번호 앞 3자리", "941")
        try:
            zip_prefix = int(zip_prefix)
        except ValueError:
            st.warning("우편번호는 숫자만 입력하세요.")
            zip_prefix = 0
        # threshold = st.slider("판단 기준 (Threshold)", 0.0, 1.0, 0.4, 0.05)
        threshold = 0.7

        submitted = st.form_submit_button("예측하기")

    if submitted:
        try:
            expected_features = model.booster_.feature_name()
            input_dict = {f: 0 for f in expected_features}

            # 입력 반영
            input_dict["amount_requested"] = amount_requested
            input_dict["employment_length"] = employment_length
            input_dict["dti"] = dti
            input_dict["zip_prefix"] = zip_prefix

            now = datetime.now()
            input_dict["issue_year"] = now.year
            input_dict["issue_month"] = now.month

            state_col = f"state_{state}"
            if state_col in input_dict:
                input_dict[state_col] = 1

            input_df = pd.DataFrame([input_dict])[expected_features]

            # 예측
            proba = model.predict_proba(input_df)[:, 1][0]
            raw_pred = int(proba >= threshold)
            prediction = 1 - raw_pred  # 승인=0, 거절=1 맞추기

            # 출력
            st.subheader("📌 예측 결과")
            st.write(f"Prediction: {'👌 승인(0)' if prediction == 0 else '🙅‍♀️ 거절(1)'}")
            st.metric(label="Probability", value=f"{proba:.2f}", delta=f"Threshold={threshold:.2f}")

            # 로그 저장
            log_data = {
                "timestamp": now,
                "amount_requested": amount_requested,
                "employment_length": employment_length,
                "dti": dti,
                "state": state,
                "zip_prefix": zip_prefix,
                "prediction": prediction,
                "probability": proba
            }
            df_log = pd.DataFrame([log_data])
            if os.path.exists(LOG_PATH):
                df_log.to_csv(LOG_PATH, mode="a", header=False, index=False)
            else:
                df_log.to_csv(LOG_PATH, index=False)

        except Exception as e:
            st.error(f"예측 실패: {e}")

# 모니터링
elif menu == "📊 모니터링":
    st.title("📊 예측 모니터링 대시보드")

    if not os.path.exists(LOG_PATH):
        st.warning(f"{LOG_PATH} 파일이 없습니다. 아직 예측 요청이 기록되지 않았습니다.")
    else:
        df = pd.read_csv(LOG_PATH)

        if "prediction" not in df.columns:
            st.error("'prediction' 컬럼이 로그에 없습니다.")
        else:
            total = len(df)
            approved = (df["prediction"] == 0).sum()
            rejected = (df["prediction"] == 1).sum()

            st.subheader("모니터링 횟수")
            st.write(f"전체 요청 수: {total}")
            st.write(f"승인(0): {approved}건 ({approved/total:.2%})")
            st.write(f"거절(1): {rejected}건 ({rejected/total:.2%})")

            st.subheader("📊 승인/거절 분포")

            # prediction 컬럼을 숫자로 변환하고 0, 1만 남기기
            df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
            pred_counts = df[df["prediction"].isin([0, 1])]["prediction"].value_counts().sort_index()

            # 숫자 → 라벨 매핑
            pred_counts.index = pred_counts.index.map({0: "승인(0)", 1: "거절(1)"})

            st.bar_chart(pred_counts)

            if "state" in df.columns:
                st.subheader("🌎 주(State)별 승인율")
                state_stats = df.groupby("state")["prediction"].apply(lambda x: (x == 0).mean())
                st.bar_chart(state_stats)

            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df["date"] = df["timestamp"].dt.date
                daily_stats = df.groupby("date")["prediction"].apply(lambda x: (x == 0).mean())

                st.subheader("📅 날짜별 승인율 추세")
                st.line_chart(daily_stats)
