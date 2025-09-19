# inspect_features.py
from save_artifacts import model

# LightGBM 모델이 학습에 사용한 feature 이름 확인
features = model.booster_.feature_name()
print("총 feature 개수:", len(features))
for f in features:
    print(f)