import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset

import math

import numpy as np
import pandas as pd

import os
import os.path as osp

import matplotlib.pyplot as plt

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

def main():
    root = "/home/paokimsiwoong/workspace/github.com/paokimsiwoong/ml_practice/transformer/my_dataset"
    file_list = os.listdir(root)

    df_list = []
    for file_name in file_list:
        file_path = osp.join(root, file_name)
        df = pd.read_excel(file_path, engine='openpyxl')
        df_list.append(df.copy())

    data = pd.concat(df_list, ignore_index=True)

    print("data loading complete")

    # 1) 토크나이저 초기화
    # BPE(Byte Pair Encoding) 기반의 토큰화 모델 생성
    # unk_token="[UNK]"는 사전에 없는 단어를 만났을 때 사용하는 'unknown' 토큰 지정
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # 2) 학습용 트레이너 설정
    trainer = BpeTrainer(
        vocab_size=10000,  # 학습하고자 하는 어휘 사전 크기 (최대 토큰 개수)
        # 1만~5만 사이 값 지정
        # 소규모 실험의 경우 1만
        # 수십만 문장의 경우 2~3만
        # 수백만 문장의 경우 3~5만
        special_tokens=["[SOS]", "[EOS]", "[PAD]", "[UNK]", "[SEP]",]  # 모델에서 사용하는 특수 토큰들
    )


    # 3) 학습용 데이터 준비
    corpus = data["번역문"]

    try:

        # 4) 토크나이저 학습 실행
        # corpus를 순회하며 BPE 병합 규칙과 토큰 사전을 학습
        tokenizer.train_from_iterator(corpus, trainer)
    except Exception as e:
        print(e)


    # 5) 학습된 토크나이저로 문장 토큰화
    encoded = tokenizer.encode("How are you?")
    print(encoded.tokens)  # 토큰화 결과 출력

    dest = "/home/paokimsiwoong/workspace/github.com/paokimsiwoong/ml_practice/transformer/tokenizer_saves"

    result = osp.join(dest, "eng_tokenizer.json")

    tokenizer.save(result)


    loaded_tokenizer = Tokenizer.from_file(result)

    encoded_from_loaded = loaded_tokenizer.encode("How are you?")
    print(f"==>> encoded_from_loaded: {encoded_from_loaded.tokens}")



if __name__ == "__main__":
    main()
