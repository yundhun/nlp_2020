import json
import unidecode
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

pst = PorterStemmer()
vocab = set()
y_label = set()
#불용어 사전
stop_words = set(stopwords.words('english'))        

f = open('./friends_train.json') 
fr_tr = json.load(f)

for i in fr_tr:
    for j in i:
        txt = j['utterance']
        emo = j['emotion']
        #텍스트에서 유니코드 제거
        txt = unidecode.unidecode(txt)
        #대소문자 통일
        txt = txt.lower()
        #기호 제거
        txt = re.sub('[^a-zA-Z]', ' ', txt)
        #토큰화 수행
        word_token = word_tokenize(txt)
        for w in word_token:
            vocab.add(w) #모든 단어 추출 (최초 1회만 수행)
        #모든 감정 추출
        y_label.add(emo)

f2 = open('./data/EmotionLines/Friends/friends_test.json') 
fr_te = json.load(f2)

for i in fr_tr:
    for j in i:
        txt = j['utterance']
        emo = j['emotion']
        #텍스트에서 유니코드 제거
        txt = unidecode.unidecode(txt)
        #대소문자 통일
        txt = txt.lower()
        #기호 제거
        txt = re.sub('[^a-zA-Z]', ' ', txt)
        #토큰화 수행
        word_token = word_tokenize(txt)
        for w in word_token:
            vocab.add(w) #모든 단어 추출 (최초 1회만 수행)
        #모든 감정 추출
        y_label.add(emo)        

#단어에 번호 매기기
vocab_seq = {}
vocab_seq_reverse = {}
for i,x in enumerate(vocab):
    vocab_seq[x] = i+1
    vocab_seq_reverse[i+1] = x

#없는 단어의 시퀀스 추가
vocab_seq['unkown_word'] = len(vocab_seq)+1
vocab_seq_reverse[len(vocab_seq)] = 'unkown_word'

#라벨 시퀀스
y_label_seq = {}
for i,x in enumerate(y_label):
    y_label_seq[x] = i

#train data 만들기
import numpy as np
tr_x = []
tr_y = []

max_len = -1

for i in fr_tr:
    for j in i:
        txt = j['utterance']
        emo = j['emotion']
        #텍스트에서 유니코드 제거
        txt = unidecode.unidecode(txt)
        #대소문자 통일
        txt = txt.lower()
        #기호 제거
        txt = re.sub('[^a-zA-Z]', ' ', txt)
        #토큰화 수행
        word_token = word_tokenize(txt)
        result = []
        for token in word_token:
            result.append(vocab_seq[token]+3)
        if( len(result) > max_len ):
            max_len = len(result)
        tr_x.append(result)
        tr_y.append(y_label_seq[emo])

tr_x = np.array(tr_x)
tr_y = np.array(tr_y)

te_x = []
te_y = []

for i in fr_te:
    for j in i:
        txt = j['utterance']
        emo = j['emotion']
        #텍스트에서 유니코드 제거
        txt = unidecode.unidecode(txt)
        #대소문자 통일
        txt = txt.lower()
        #기호 제거
        txt = re.sub('[^a-zA-Z]', ' ', txt)
        #토큰화 수행
        word_token = word_tokenize(txt)    
        result = []
        for token in word_token:
            if( token in vocab_seq ):
                result.append(vocab_seq[token]+3)
            else:
                result.append(len(vocab_seq)+3)
        te_x.append(result)
        te_y.append(y_label_seq[emo])

te_x = np.array(te_x)
te_y = np.array(te_y)

import pickle
with open('./friends.p', 'wb') as file:    # james.p 파일을 바이너리 쓰기 모드(wb)로 열기
    pickle.dump(tr_x, file)
    pickle.dump(tr_y, file)
    pickle.dump(te_x, file)
    pickle.dump(te_y, file)
    pickle.dump(vocab_seq,file)

