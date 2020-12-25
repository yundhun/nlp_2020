자연어처리 프로젝트 기말과제
윤대훈/2019511016/빅데이터융합학과

**Kaggle 예측파일인 sample.csv은 Friends는 friends_emotion폴더, 네이버영화리뷰는 naver_move폴더에 있습니다.** 

**각 폴더에 대한 설명은 아래 참고**

# Friends 대사 데이터 분석 코드 
**폴더 : friends_emotion**

**접근 방법**: BERT 모델을 활용하여 감정분류 모델 수립. 
클래스가 불균형 함으로 이를 해결하기 위해 클래스별 Weight 를 조정하여 학습함.
예측력을 높이기 위해 Droupout 을 조정하여 학습함.
>  실행 방법 : 동일 폴더 내에 Friends 데이터셋 저장 후 ipynb 순차 실행

**참조코드**: 자연어처리 수업 강의자료 7번 예제 코드

**참조 코드 수정 사항**
- Label Imbalance 문제 해결을 위한 Crossentrophy Weight 계산 로직 추가
- Droupout 0.5로 조정
- 대사가 두문장으로 이루어져 있는 경우 [SEP] 토큰 추가

# Naver 영화리뷰 데이터 분석 코드
**폴더 : naver_movie**
**접근 방법**: 한국어 BERT 모델을 활용하여 감정분류 모델 수립. 
예측력을 높이기 위해 Droupout 을 조정하여 학습함.
> 실행 방법 : 동일 폴더 내에 Naver 영화 리뷰 데이터셋을 저장 후 ipynb 순차 실행
> Validation Data 는 반드시  kaggle 의 csv 가 아닌 동일 폴더에 있는 valid.txt 사용부탁드립니다.

**참조 코드**: https://github.com/SKTBrain/KoBERT/blob/master/scripts/NSMC/naver_review_classifications_pytorch_kobert.ipynb

**참조 코드 수정 사항**
- 특별히 없음 

# Final Report 개발에 사용된 코드
**폴더 : final_report**

*반드시 동일 폴더에 friends_train.json, friends_test.json, friends_dev.json 을 다운로드 받은 후 실행
*pretrained embedding 사용을 위해 glove.6B.50d.txt ( https://www.kaggle.com/watts2/glove6b50dtxt 에서 다운로드 )를 /glove 폴더에 저장 후 실행

**내용:**
- **Friends 데이터 전처리 코드**
>실행방법 :  python preprocessing.py

- **A Structured Self-Attentive Sentence Embedding 모델 학습 코드**
>실행방법 : python classification.py "friends"
>실행후 /visualization/attention.html 에 결과 파일 생성

