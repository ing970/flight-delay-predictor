# flight-delay-predictor
<img width="369" alt="img" src="https://github.com/ing970/flight-delay-predictor/assets/70427747/73555579-1bad-4cc7-b540-25e261606456">

미국 항공기 연착 예측을 위한 Streamlit 웹 애플리케이션입니다.
- https://flight-delay-predictor.streamlit.app/

## 주요 특징:
- XGBoost 모델을 사용하여 연착을 예측합니다.
- 공항 코드를 기반으로 해당하는 주(State)의 정보를 제공합니다.

## 사용 예시:
![사용방법](https://github.com/ing970/flight-delay-predictor/assets/120775224/adf459ee-59c0-436a-8e67-2206ddbba18d)


## 사용 방법:
1. 필요한 패키지를 설치합니다: `pip install streamlit xgboost joblib`
2. 애플리케이션을 실행합니다: `streamlit run main.py`

## 데이터 출처:
본 프로젝트에서 사용된 데이터는 Dacon의 "월간 데이콘 항공편 지연 예측 AI 경진대회"에서 제공된 데이터를 활용하였습니다.
- https://dacon.io/competitions/official/236094/overview/description


## Contents:
1. **Data Preprocessing**: 데이터 전처리 과정은 `0_preprocessing.ipynb` 파일에서 확인 가능합니다.
    - 원본 데이터 출처: 월간 데이콘 항공편 지연 예측 AI 경진대회
    - 원활한 EDA를 위한 초기 전처리 단계 포함
    - `distance_dict.pkl` 및 `time_dict.pkl`: 전처리에 사용된 추가 데이터 및 유틸리티
  
2. **Exploratory Data Analysis (EDA)**: 탐색적 데이터 분석 과정은 `1_EDA.ipynb`에서 확인할 수 있습니다.
    - Label 불균형 분석
    - 항공사별 지연 비율 분석
      
3. **Modeling**: 모델링 과정은 `2_model.ipynb`에서 살펴볼 수 있습니다.
    - 사용한 모델: XGBoost
    - 범주형 변수의 label encoding을 활용하여 모델 학습
    - `label_classes.pickle`: 레이블 클래스 정보가 포함된 파일
      
4. **Prediction Application**: `main.py`를 통해 항공편 지연 예측 애플리케이션을 실행할 수 있습니다.
    - 사용된 플랫폼: Streamlit
    - 공항 코드와 전체 이름에 대한 매핑 정보 포함
