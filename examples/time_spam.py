import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_time_spam_correlation(df):
    # 날짜를 datetime으로 변환
    df['날짜'] = pd.to_datetime(df['날짜'])
    
    # 시간대 추출 (0-23)
    df['시간대'] = df['날짜'].dt.hour
    
    # 메일종류를 숫자로 변환 (스팸 여부)
    df['스팸여부'] = df['메일종류'].map({'햄': 0, '스팸': 1})
    
    # 시간대별 스팸 비율 계산
    pivot_table = df.pivot_table(values='스팸여부', index='시간대', aggfunc='mean')
    
    # 히트맵 시각화
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table.T, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('시간대별 스팸 메일 비율')
    plt.xlabel('시간대')
    plt.ylabel('스팸 비율')
    plt.show()

# 사용 예시:
df = pd.read_csv('Rabble/datasets/combined_data.csv')  # 데이터셋 로드
analyze_time_spam_correlation(df)  # 히트맵 분석