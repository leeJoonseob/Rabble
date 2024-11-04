#====================================================================================
# Morphological Analysis
# 형태소 분석
#====================================================================================

#형태소 분석 자연 언어 처리에서 말하는 형태소 분석이란 어떤 대상 어절을 최소의 의미 단위인 '형태소'로 분석하는 것을 의미한다. 
#(형태소는 단어 그 자체가 될 수도 있고, 일반적으로는 단어보다 작은 단위이다.)

#In natural language processing, morphological analysis refers to the process of analyzing a target word or phrase 
#into its smallest meaningful units called 'morphemes'.
#(Morphemes can be words themselves, but generally, they are smaller units than words.)
from konlpy.tag import Kkma
from konlpy.tag import Hannanum
from konlpy.tag import Komoran 
from konlpy.tag import Okt
from kiwipiepy import Kiwi


kiwi = Kiwi()
hannanum = Hannanum()
kkma = Kkma()
komoran = Komoran()
okt = Okt()

# #sample text
# text = """엘지유플러스 재약정안내문 [Web발신](광고)우수고객님 감사합니다.
#             안녕하십니까~엘지 유플러스고객 행복지점입니다. 
#             ☏ 상담번호 ☏☞070)4499-3293"""

# print("Hannanum 형태소 분석: \n", hannanum.morphs(text))
# print("==============================")
# print("Kkma 형태소 분석: \n", kkma.morphs(text))
# print("==============================")
# print("Komoran 형태소 분석: \n", komoran.morphs(text))
# print("==============================")
# print("Okt 형태소 분석: \n", okt.morphs(text))
# print("==============================")
# for reslut in kiwi.analyze(text):
#     print(reslut)
