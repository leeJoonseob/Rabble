import pandas as pd
import logging
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("import 완료")

# load data
spam_ham = pd.read_csv("Rabble/datasets/combined_data.csv")
logger.info("데이터 로드 완료")

# change data type to numeric
spam_ham['메일종류'] = spam_ham['메일종류'].map({'햄': 0, '스팸': 1})
logger.info("데이터 타입 변경 완료")

train = spam_ham['메일제목']
label = pd.DataFrame(spam_ham['메일종류'])

logger.info("데이터 분리 완료")

print(spam_ham.shape)

# 클래스 분포 시각화
plt.figure(figsize=(12, 5))

# 막대 그래프
plt.subplot(1, 2, 1)
sns.set_theme(style="darkgrid")
ax = sns.countplot(x='메일종류', data=spam_ham)
ax.set_xticklabels(['Ham', 'Spam'])
ax.set_title('Class Distribution (Bar Plot)')
ax.set_xlabel('Mail Type')
ax.set_ylabel('Count')

# 각 막대 위에 개수 표시 (수정된 부분)
for i, v in enumerate(spam_ham['메일종류'].value_counts()):
    ax.text(i, v, str(v), ha='center', va='bottom')

# 파이 차트
plt.subplot(1, 2, 2)
class_counts = spam_ham['메일종류'].value_counts()
plt.pie(class_counts, labels=['Ham', 'Spam'], autopct='%1.1f%%', startangle=90)
plt.title('Class Distribution (Pie Chart)')

plt.tight_layout()
plt.show()

# 클래스 비율 출력
total = len(spam_ham)
ham_ratio = class_counts[0] / total * 100
spam_ratio = class_counts[1] / total * 100

print(f"Ham 메일 비율: {ham_ratio:.2f}%")
print(f"Spam 메일 비율: {spam_ratio:.2f}%")