import pandas as pd

def combine_data():
    # 학습 데이터 로드
    google_message = pd.read_csv("Rabble/datasets/messages.csv")
    spam_message = pd.read_csv("Rabble/datasets/spam_virus.csv")
    ham_message = pd.read_csv("Rabble/datasets/msg.csv")
    #spam_msg = pd.read_csv("Rabble/datasets/spam_dataset.csv")
    
    # 메일 종류 추가
    google_message['메일종류'] = '햄'
    spam_message['메일종류'] = '스팸'
    ham_message['메일종류'] = '햄'
    #spam_msg['메일종류'] = '스팸'

    # 필요한 열 선택
    final_google_df = google_message[['메일종류', '메일제목']]
    final_spam_df = spam_message[['메일종류', '메일제목']]

    ham_message['메일제목'] = ham_message['메일제목'] + ham_message['메일내용']
    final_ham_df = ham_message[['메일종류', '메일제목']]
    #final_spam_msg_df = spam_msg[['메일종류', '메일제목']]



    # Ham 데이터 수
    ham_count = final_google_df.shape[0] + final_ham_df.shape[0]

    # Spam 데이터 언더샘플링
    final_spam_df = final_spam_df.sample(n=ham_count, random_state=42)
    
    # 데이터 결합
    combined_df = pd.concat([final_google_df, final_spam_df,final_ham_df], axis=0, ignore_index=True)
    print(combined_df['메일종류'].value_counts())


    combined_df.to_csv('/Users/PROTEIN/Desktop/Coll_Third/spam_finder/Rabble/datasets/combined_data_under.csv', index=False, encoding='utf-8')
combine_data()