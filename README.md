# Korean → English Translation Transformer Experiments

한국어 → 영어 번역 트랜스포머를 직접 구현하고, 데이터 규모·토크나이저·학습 전략 등을 바꾸어가며 성능을 비교한 실험 결과를 정리한 저장소

모델 구현은 Harvard NLP의 Annotated Transformer 코드를 기반으로 하며, 실험 과정을 간단하게 여기에 정리함

> #### 모델 코드 기반  
> - Annotated Transformer: https://nlp.seas.harvard.edu/annotated-transformer/
> #### 사용 데이터셋
> - AIHUB 한국어-영어 번역(병렬) 말뭉치: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=126
> - nayohan/aihub-en-ko-translation-12m: https://huggingface.co/datasets/nayohan/aihub-en-ko-translation-12m

---

## 1. 데이터셋 및 전처리

### 1-1. 1M 데이터셋 전처리

1M 규모 데이터셋(AIHUB 한국어-영어 번역(병렬) 말뭉치)은 **분류별로 나뉜 여러 xlsx 파일을 하나의 csv로 합친 후, `kor`, `en`, `cat` 세 열만 남기는 방식**으로 구축

- 여러 개의 xlsx 파일을 로드 후 세로 방향으로 concat
- 불필요한 메타데이터 열 제거
- `kor` (한국어 문장), `en` (영어 번역 문장), `cat` (문장 분류) 열만 남김
- NaN, 비정상 문자열 필터링


| column   |   num_examples |   avg_len |   min_len |   max_len |   p50_len |   p90_len |   p95_len |   p99_len |
|:---------|---------------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|
| kor      |        1602418 |   57.5784 |         4 |       359 |        59 |        95 |       105 |       123 |
| en       |        1602418 |  141.7    |        10 |       999 |       137 |       251 |       285 |       358 |

| 분류 | 개수 |
|:------------:|:--------:|
| 문어쳬_뉴스 | 801387 |
| 구어체 | 400000 |
| 문어체_한국문화 | 100646 |
| 문어체_조례 | 100298 |
| 문어체_지자체웹사이트 | 100087 |
| 대화체 | 100000 |

![한국어 문장 길이](imgs/1m_kor.png)
![영어 문장 길이](imgs/1m_en.png)


### 1-2. 10M 데이터셋 전처리 (12M → 10M 축소)

[ayohan/aihub-en-ko-translation-12m 데이터셋](https://huggingface.co/datasets/nayohan/aihub-en-ko-translation-12m)을 전처리를 통해 10M 규모로 정제

#### 1-2-1. 세부 분류 통합 (수백 종 → 8개 대분류)

12M 데이터셋은 여러 데이터셋이 합쳐지면서 **문장 분류(category)가 수백 종(domain 664개, subdomain 657개)**로 나누어져 있어 정리가 필요함

![domain](imgs/12m_domain.png)
![subdomain](imgs/12m_subdomain.png)


예시:
- "문화", "문화·예술", "문화·교육", "문화재", "민속", "생활·민속", "문화유산", "역사", "역사/근현대", "역사/전통 시대" 등 위와 같은 라벨들을 **단일 대분류 `"문화/예술/역사"`**로 통합

예시와 같이 유사한 문장 분류들을 8개 대분류로 통합
- 유사/중복 라벨들을 규칙 기반 매핑 테이블로 통합
- 최종적으로 약 8개 수준의 상위 카테고리로 정리


| 분류 | 문장 수 |
|:-----------------:|:--------------------:|
| 과학/기술/학술자료 | 3,551,719 |
| 일상/대화 | 2,913,310 |
| 문화/예술/역사 | 1,260,217 |
| 뉴스/시사 | 1,181,603 |
| 의학/보건 | 653,544 |
| 법률/행정 | 621,916 |
| 특허 | 358,307 |
| 금융/경제 | 155,400 |


> domain, subdomain 값이 None이라서 8개 분류로 정리되지 않고 기타로 분류된 1501722개의 데이터를 확인해보니 1M 데이터셋의 데이터와 완전 동일 ==> 12M 데이터셋에 1M 데이터셋이 포함되어 있었음 
>> 1M 데이터셋의 1602418개의 데이터중 100646개인 문어체_한국문화를 제외한 1501722개가 domain, subdomain None으로 12M 데이터셋에 포함되어있음  
>> --> 1M 데이터셋 분류가 구어체, 대화체인 데이터는 일상/대화로 분류하고 문어체_뉴스, 문어체_지자체웹사이트는 뉴스/시사, 문어체_조례는 법률/행정로 재분류  
>>  문어체_한국문화 100646개는 domain, subdomain이 100158개는 문화/예술/역사 계열, 477개는 뉴스/시사 계열, 11개는 법률/행정 계열로 포함되어 있어 그대로 둠

#### 1-2-2. 문어체/구어체/혼재 대분류(style)

분류 이름에 문어체, 구어체 표시가 되어 있는 1M 데이터셋과 다르게 별도의 style 열에 문어체, 구어체 여부를 저장하고 있어 추후 활용을 위해 남겨두기로 결정

![style](imgs/12m_style.png)

하지만 상당수의 데이터는 None으로 표시되어 있어 결측치를 적절한 값으로 채워줄 필요가 발생

![분류별 None 값 개수](imgs/12m_none_style_data_per_cat.png)

None style인 데이터가 위에서 통합한 8개 분류 중 문화/예술/역사인 경우 문어체로 된 역사, 정치등의 자료도 있지만,  
방송콘텐츠, 전통문화, 민속, 구비 전승 등 구어체를 포함하는 자료도 있으므로 style 값을 문어체/구어체 혼재로 설정

일상/대화는 구어체, 특허는 특허로 두고 나머지 분류들은 전부 문어체로 설정


| style | 문장 수 |
|:-----------------:|:--------------------:|
| 문어체 | 6,555,552 |
| 구어체 | 2,938,318 |
| 문어체/구어체 혼재 | 888,669 |
| 특허 | 358,307 |


#### 1-2-3. 중복 및 상충 번역 제거

여러 데이터셋을 합치면서 다음과 같은 중복/충돌 케이스가 발생

- (1) **한국어–영어 문장 쌍이 완전히 동일**한 경우: 완전 중복 → 하나만 유지
- (2) **한국어는 같고 영어 번역이 다른 경우**: 상충 번역 → 학습에 혼란을 줄 수 있으므로 제거

처리 전략:

- (kor, en) 쌍 기준으로 완전 중복 제거
- 동일 kor에 대해 서로 다른 en 후보가 존재하는 경우: 전체에서 제거

> TODO: 중복/충돌 제거 전/후 데이터 수 비교 표 추가

#### 1-2-4. NaN 및 이상값 처리

- `kor`, `en` 중 하나라도 NaN이면 해당 행 제거
- TODO: 지나치게 짧은 문장(토큰 수 1 이하), 지나치게 긴 문장(토큰 수 256~512) 또는 비문자열 타입 값 제거

> TODO: NaN/이상값 필터링 전/후 통계 표 추가

#### 1-2-5. 1M 테스트셋 중복 제거

실험 결과 비교를 위해 **1M, 10M 모두 동일한 test set을 사용**

- 1M 데이터셋에서 test split으로 사용되는 `(kor, en)` 쌍을 수집
- 10M 데이터셋에서 동일한 `(kor, en)`이 등장하는 행을 전부 제거
- 이렇게 만들어진 10M train/dev에서는 1M test와의 데이터 누수를 방지

> TODO: 1M test와 10M train 간 교집합 개수(제거 전) / 제거 후 결과 보고  
> TODO: 1M vs 10M 데이터셋에서 동일 test set 기준 BLEU/METEOR/ChrF 비교 그래프 자리 (WandB 스크린샷)

---

## 2. 평가 메트릭

번역 성능 평가는 다음 세 가지 메트릭으로 수행함

- **BLEU**  
  - 전통적인 n-gram 기반 번역 평가 지표
  - 직관적인 비교에 유용하지만, 의미적 동등성을 완벽히 반영하지는 못함
- **METEOR**  
  - 스템·동의어 등을 고려하여 BLEU보다 의미적 유사성을 더 잘 반영
- **ChrF**  
  - 문자 단위 F-score 기반 지표로, 특히 한국어/영어 같은 형태소 구조가 다른 언어 쌍에서 유용

> TODO: 각 실험 조건별 BLEU / METEOR / ChrF 결과 표  
> TODO: 각 메트릭에 대한 WandB 라인 플롯 이미지 자리

---

## 3. 토크나이저 비교 및 최종 선정

두 종류의 한국어–영어 대응 토크나이저를 비교하여 최종 토크나이저를 선정

### 후보 토크나이저

- `KETI-AIR/ke-t5-base`
- `Translation-EnKo/exaone3-instrucTrans-v2-enko-7.8b`

### 커버리지와 학습 안정성

- `Translation-EnKo/exaone3-instrucTrans-v2-enko-7.8b`
  - [nayohan/aihub-en-ko-translation-12m](https://huggingface.co/datasets/nayohan/aihub-en-ko-translation-12m) 데이터셋을 사용해 학습한 모델의 토크나이저
  - 1M, 10M 데이터셋 모두에서 **unknown 토큰 없이 전체 문장을 토큰화 가능**  
  - 단, vocabulary 크기가 매우 큼 (102,400) → embedding/softmax 차원이 커져 학습이 잘 진행되지 않고, 메모리 사용량도 커짐
- `KETI-AIR/ke-t5-base`  
  - vocab size: 64,100  
  - 학습이 안정적으로 진행되며, 수렴도 잘 되는 편  
  - 단, 일부 토큰이 `<unk>`으로 처리:
    - 1M train 전체 토큰 중 약 0.05%가 unknown
    - 10M train 전체 토큰 중 약 0.08%가 unknown

### 최종 선택

- **최종 토크나이저: `KETI-AIR/ke-t5-base`**
  - 이유:
    - Vocab 크기가 적당하여 학습이 안정적
    - 소량의 unknown 토큰 비율은 무시 가능한 수준
    - 메모리 사용량과 연산량을 고려할 때 가장 실용적

> TODO: 각 토크나이저별  
> - vocab size,  
> - unknown 비율,  
> - 학습 곡선 (WandB),  
> - 최종 BLEU/METEOR/ChrF  
> 를 비교하는 표 및 그래프 자리

>> TODO: 10m 데이터셋에서는 결과가 달라지는지 확인 필요

---

## 4. Embedding / Generator Weight Tying 실험

Transformer 구조에서 다음 세 weight 간의 tying 여부에 따른 성능을 비교

- Encoder embedding (입력 임베딩)
- Decoder embedding (출력 임베딩)
- Generator weights (decoder output → vocab logits projection)

### 실험 조건

1. **완전 공유 (3-way tying)**  
   - encoder embedding = decoder embedding = generator weights
2. **encoder embedding = decoder embedding (2-way tying)**
3. **encoder embedding = generator weights (2-way tying)**
4. **decoder embedding = generator weights (2-way tying)**
5. **weight tying 미사용 (모두 분리)**

각 설정에 대해 동일한 데이터/하이퍼파라미터로 학습 후 번역 성능을 비교함

### 최종 결과

- **weight tying 미사용 (5번)가 가장 좋은 성능**
  - tying을 통해 파라미터 수는 줄어들지만, 한국어-영어 번역이라는 비대칭 작업에서는 자유도가 높은 설정이 더 유리

> TODO: 5가지 설정별 파라미터 수 / BLEU / METEOR / ChrF 비교 표  
> TODO: WandB 실험 페이지 링크 및 스크린샷 자리

---

## 5. Learning Rate 스케줄 실험

학습률 스케줄은 **Annotated Transformer 글에서 제안된 함수**를 기본으로 사용하되, 여러 변형을 실험함
### 기본 스케줄 (Annotated Transformer 스타일)

- 형식:  
  $$
  lr = d_{model}^{-0.5} \cdot \min\left(step^{-0.5}, step \cdot warmup^{-1.5}\right)
  $$
- 실험한 변수:
  - warmup step 수 (예: 4,000 / 8,000 / 16,000 등)
  - 전체 스케일링 factor (multiplying factor) 변경

### 추가 스케줄

- 단순 감쇠 함수 (예: \(a^{-x}\) 꼴의 지수 감쇠)
- step/epoch 기반 cosine decay 등의 변형도 비교 가능하도록 코드 구조화

각 스케줄에 대해 수렴 속도, 최종 성능, 안정성(gradient explosion/vanishing 여부)을 관찰

> TODO:  
> - step에 따른 lr 변화 그래프 (WandB)  
> - 각 스케줄별 학습 곡선 / 최종 메트릭 비교 그래프  
> - lr 스케줄별 best checkpoint 성능 표

---

## 6. Batch Size 및 Gradient Accumulation 실험

### 6-1. Batch Size와 성능

- 초기 실험: batch size 8 → 16 → 32로 증가  
  - **batch size가 커질수록 성능이 지속적으로 향상**됨을 확인
- GPU 메모리 한계로 인해 batch size를 64 이상으로 직접 올리기 어려워, **gradient accumulation** 도입
  - 예: 실제 batch size 64, 128, 256에 대응하는 effective batch를  
    - base batch 32 + accumulation steps 2, 4, 8 등으로 구성
  - **accumulation step이 늘어날수록 성능이 증가하는 경향** 확인

> TODO: batch size / accumulation step vs BLEU 등의 그래프 (WandB)  
> TODO: 메모리 사용량 vs batch size 비교 표/그래프 자리

---

### 6-2. Batch Size 실험에 따른 메모리 문제 해결 과정

#### 6-2-1. Mixed Precision 도입

- PyTorch AMP (`torch.cuda.amp`)를 사용해 half precision으로 계산  
  - 메모리 사용량 감소  
  - 연산 속도 증가

#### 6-2-2. Sequence 길이 변화에 따른 reserved memory 증가 문제

- 각 batch마다 sequence 길이가 달라질 때, PyTorch가 reserved memory를 계속 늘리는 문제를 관찰
- 해결: **memory pre-allocation** 전략 도입
  - 고정된 최대 토큰 수를 기준으로 메모리를 미리 할당
  - 매 batch에서 이 범위 내에서만 연산하도록 조정

##### 6-2-2-1. Custom Sampler 시도

- 아이디어: batch마다 `token + pad` 개수를 일정하게 유지
  - TODO: 토큰 개수 기준으로 bucket을 만들고, 비슷한 길이 문장끼리 묶어서 배치 구성
- 하지만, 실제 실험에서는 다음 전략이 더 효율적이었음
  - **batch당 문장 개수를 일정하게 유지** 하고 memory pre-allocation 시 `batch_size × (데이터셋 전체에서의 최대 토큰 길이)` 기준으로 할당

##### 6-2-2-2. 메모리 한계 초과 시 동적 처리

batch size가 커지면서,  
`batch_size × max_seq_len` 기준으로 pre-allocation이 불가능한 경우가 발생함

해결 전략:

1. 가능한 범위에서 최대한 크게 memory pre-allocation 수행
2. 해당 범위를 초과하는 규모의 배치가 들어오는 경우:
   - `gc.collect()`, `torch.cuda.empty_cache()`로 메모리 수동 정리
   - 공유 메모리까지 활용하여 해당 배치를 처리
3. 해당 배치가 끝나면:
   - 다시 `gc.collect()`, `torch.cuda.empty_cache()`로 정리
   - 기존 크기로 memory pre-allocation 재수행

> TODO:  
> - pre-allocation 유무에 따른 메모리 사용량/속도 비교 그래프  
> - 길이 분포 / bucket 전략 설명 그림

#### 6-2-3. Gradient Checkpoint 도입

- vocab 크기가 큰 토크나이저(EXAONE 계열)를 사용할 때는  
  각 Transformer encoder, decoder block에 **gradient checkpoint**를 적용하여 메모리 사용량을 추가로 줄임
- 구현:
  - `torch.utils.checkpoint.(m, out, mask, testing, use_reentrant=False)`를 encoder/decoder block에 적용
- 효과:
  - 피크 메모리 사용량 감소
  - 계산량 증가(속도 저하)는 있지만, 더 큰 batch/effective batch로 학습 가능

> TODO:  
> - checkpoint 사용 전/후 메모리 사용량 비교 그래프  
> - checkpoint 사용 전/후 학습 속도 vs 성능 트레이드오프 그래프

---

## 7. WandB 로그 및 결과 정리 (Placeholder)

본 저장소의 실험 결과는 대부분 WandB를 통해 기록함

- 프로젝트 이름: `transformer 한영 번역 실험`

> TODO: 아래 항목에 WandB 링크/이미지 삽입
>
> - [ ] 데이터셋 규모(1M vs 10M)에 따른 성능 비교 그래프  
> - [ ] 토크나이저별 학습 곡선 및 최종 메트릭  
> - [ ] weight tying 설정별 성능 비교  
> - [ ] learning rate 스케줄별 학습/검증 loss 및 BLEU  
> - [ ] batch size / gradient accumulation에 따른 성능 및 메모리 사용량

---

## 8. 재현 방법 (예시)

```bash
# 0. 환경 설치

# 1. 데이터 전처리

# 2. 1M 데이터셋 + 기본 설정 학습

# 3. 10M 데이터셋 + gradient accumulation + best tokenizer 설정

```

---

## 9. 참고 자료
- Annotated Transformer 구현: https://nlp.seas.harvard.edu/annotated-transformer/
- Hugging Face Datasets / Tokenizers 문서: https://huggingface.co/docs/datasets
- PyTorch gradient checkpointing: https://pytorch.org/docs/stable/checkpoint.html