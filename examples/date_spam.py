import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def analyze_date_spam_heatmap(df):
    # 날짜를 숫자로 변환 (타임스탬프)
    df['날짜'] = pd.to_datetime(df['날짜']).astype(int) // 10**9
    
    # 메일종류를 숫자로 변환 (스팸 여부)
    le = LabelEncoder()
    df['메일종류'] = le.fit_transform(df['메일종류'])
    
    # 날짜와 메일종류만 선택
    correlation_data = df[['날짜', '메일종류']]
    
    # 상관계수 계산
    correlation_matrix = correlation_data.corr() 
    
    # 히트맵 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap Between Date and Spam Classification')
    plt.show()

# 사용 예시:
df = pd.read_csv('Rabble/datasets/combined_data.csv' )  # 데이터셋 로드
analyze_date_spam_heatmap(df)  # 히트맵 분석