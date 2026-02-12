import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Sampler, get_worker_info
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict, ClassLabel

# SacreBLEU 등 메트릭을 제공하는 huggingface의 라이브러리
import evaluate

import multiprocessing

from functools import partial
# 함수 일부 인자 고정에 사용

from collections import defaultdict
# custom sampler에서 사용

class Loaders():
    def __init__(
            self,
            data_path="/home/paokimsiwoong/workspace/github.com/paokimsiwoong/ml_practice/transformer/data.csv",
            max_token_length = 512,
            target_tokens = 2000,
            batch_size_train = 8,
            num_workers = 4,
            batch_size_val = 4,
            batch_size_test = 4,
            val_num_workers = 4,
            # start_idx = 64100, 
            # end_idx = 1, 
            # padding_idx = 0, 
            # unk_idx = 2,
            tokenizer = "KETI-AIR/ke-t5-base",
            seed = 42,
    ):
        # 1) 데이터셋 로드
        dataset = load_dataset("csv", data_files=data_path)['train']
        # AI hub 한국어-영어 번역(병렬) 말뭉치 데이터셋
        # kor, en, cat 3개의 칼럼으로 구성

        # 2) 카테고리 컬럼의 고유 클래스 찾아서 ClassLabel 객체 생성
        unique_classes = dataset.unique('cat')  
        # 'cat' 열에 문장 분류 
        # 0: 구어체
        # 1: 대화체
        # 2: 문어체_뉴스
        # 3: 문어체_한국문화
        # 4: 문어체_조례
        # 5: 문어체_지자체웹사이트

        print(f"==>> unique_classes: {unique_classes}")
        class_label = ClassLabel(names=unique_classes)

        # 3) 기존 컬럼 타입 변경 (캐스팅)
        dataset = dataset.cast_column('cat', class_label)

        # 4) stratify_by_column으로 분할
        train_validtest = dataset.train_test_split(test_size=0.2, seed=seed, stratify_by_column='cat')
        valid_test = train_validtest['test'].train_test_split(test_size=0.1, seed=seed, stratify_by_column='cat')

        dataset_dict = DatasetDict({
            'train': train_validtest['train'],
            'validation': valid_test['train'],
            'test': valid_test['test']
        })

        # print(dataset_dict)

        NUM_CPU = multiprocessing.cpu_count()
        # print(f"==>> NUM_CPU: {NUM_CPU}")

        # self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
        # @@@ 점포, 만료 등의 단어가 <unk>인 문제 발견 => 다른 토크나이저 사용?
        # self.tokenizer = AutoTokenizer.from_pretrained("KETI-AIR/ke-t5-base")

        # special_tokens_dict = {'bos_token': '<s>'}
        # self.tokenizer.add_special_tokens(special_tokens_dict)

        self.tokenizer = None
        self.len_vocab = None
        self.start_idx = None
        self.end_idx = None
        self.padding_idx = None
        self.unk_idx = None

        self.init_tokenizer(tokenizer)

        print(f"==>> self.tokenizer name: {tokenizer}")

        print(f"==>> self.tokenizer.model_max_length: {self.tokenizer.model_max_length}")

        print(f"==>> len(self.tokenizer): {len(self.tokenizer)}")

        self.max_token_length = min(max_token_length, self.tokenizer.model_max_length)

        # partial을 이용해 tokenizer, max_token_length 인자 고정
        cetf = partial(convert_examples_to_features, tokenizer=self.tokenizer, max_token_length=self.max_token_length)
        # @@@ convert_examples_to_features함수에서 examples가 첫번쨰 인자가 아니면 
        # @@@ partial과 dataset_dict.map의 인자 전달 방식이 충돌해서 
        # @@@ TypeError: convert_examples_to_features() got multiple values for argument 'tokenizer' 에러 발생

        self.datasets = dataset_dict.map(
                                cetf,
                                # lambda examples: convert_examples_to_features(examples, tokenizer=self.tokenizer, max_token_length=self.max_token_length),
                                batched=True,
                                # 원 데이터 'en', 'kor', 'cat' 등의 칼럼을 지우려면
                                # remove_columns 인자 사용
                                # remove_columns=dataset_dict["train"].column_names,
                                num_proc=NUM_CPU)
        
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@ 길이별 bucketing을 하기 위해 length 칼럼 추가

        def add_length(batch):
            batch["length"] = [len(inp) for inp in batch["input_ids"]] 
            # @@@ map함수에 batched=True여야함
            # @@@ False이면 batch["length"] = len(batch["input_ids"])
            return batch
        self.datasets = self.datasets.map(add_length, batched=True, num_proc=NUM_CPU)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        print(f"==>> self.datasets: {self.datasets}")

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@ 길이 별 bucketing 안할 경우
        # sampler = TokenBatchSampler(
        #     self.datasets["train"], 
        #     target_tokens=10000,
        #     max_batch_samples=batch_size_train * 2,
        # )

        # @@@ 길이 기준 sort로 간이 길이 별 bucketing 실행
        self.datasets["train"] = self.datasets["train"].sort("length")

        sampler = SortedTokenBatchSampler(
            self.datasets["train"], 
            target_tokens=target_tokens,
            max_batch_samples=batch_size_train, # TODO: 적절한 값 찾기
        )
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        self.train_set = self.datasets['train']
        self.val_set = self.datasets['validation']
        self.test_set = self.datasets['test']

        c_fn = partial(collate_fn, start_idx=self.start_idx, end_idx=self.end_idx, padding_idx=self.padding_idx, unk_idx=self.unk_idx)

        # self.loader_train = DataLoader(self.train_set, batch_size=batch_size_train, collate_fn=c_fn, shuffle=True, num_workers=num_workers, pin_memory=True)
        # @@@ 배치 별 총 토큰 수 일정하게 유지하기 위해 batch_size 대신 batch_sampler 사용
        self.loader_train = DataLoader(self.train_set, batch_sampler=sampler, collate_fn=c_fn, shuffle=False, num_workers=num_workers, pin_memory=True)
        # 학습시에만 shuffle=True
        self.loader_val = DataLoader(self.val_set, batch_size=batch_size_val, collate_fn=c_fn, shuffle=False, num_workers=val_num_workers, pin_memory=True)
        self.loader_test = DataLoader(self.test_set, batch_size=batch_size_test, collate_fn=c_fn, shuffle=False, num_workers=val_num_workers, pin_memory=True)

        # BLEU 계산에 필요
        self.metric_bleu = evaluate.load("sacrebleu")
        # ChrF - 문자 단위 n-그램 기반으로, 한국어 같은 교착어 처리에 강하고 어휘 미스매치에 강건
        self.metric_chrf = evaluate.load("chrf")
        # TER (Translation Edit Rate) - 번역과 정답 간 편집 거리 기반 메트릭으로, 오류율을 직관적으로 파악할 수 있어 BLEU 보완에 유용
        # self.metric_ter = evaluate.load("ter")
        # METEOR - 단어 정렬, 동의어, 어근 등을 반영하여 인간 평가에 근접한 의미적 평가
        self.metric_meteor = evaluate.load("meteor")
        # BERTScore - 사전학습된 BERT 임베딩 기반으로 문장 수준 의미적 유사성을 평가
        # self.metric_bertscore = evaluate.load("bertscore")

        self.metric_bleu_per_cat = []
        self.metric_chrf_per_cat = []
        self.metric_meteor_per_cat = []
        # self.metric_bertscore_per_cat = []

    def init_tokenizer(self, tokenizer):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        match tokenizer:
            case "KETI-AIR/ke-t5-base":
                special_tokens_dict = {'bos_token': '<s>'}
                self.tokenizer.add_special_tokens(special_tokens_dict)

                self.len_vocab = len(self.tokenizer)
                self.start_idx = 64100
                self.end_idx = 1
                self.padding_idx = 0
                self.unk_idx = 2
            case "Translation-EnKo/exaone3-instrucTrans-v2-enko-7.8b":
                special_tokens_dict = {'pad_token': '[PAD]'}
                self.tokenizer.add_special_tokens(special_tokens_dict)

                self.len_vocab = len(self.tokenizer)
                self.start_idx = 1
                self.end_idx = 361
                self.padding_idx = 0
                self.unk_idx = 3
            case "LGAI-EXAONE/K-EXAONE-236B-A23B":
                self.len_vocab = len(self.tokenizer)
                self.start_idx = 1
                self.end_idx = 53
                self.padding_idx = 0
                self.unk_idx = 3
            case _:
                raise ValueError


    def add_batch_to_metrics(self, preds, labels):

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        self.metric_bleu.add_batch(predictions=decoded_preds, references=decoded_labels)
        self.metric_chrf.add_batch(predictions=decoded_preds, references=decoded_labels)
        # self.metric_ter.add_batch(predictions=decoded_preds, references=decoded_labels)
        self.metric_meteor.add_batch(predictions=decoded_preds, references=decoded_labels)
        # self.metric_bertscore.add_batch(predictions=decoded_preds, references=decoded_labels)

    def compute_metrics(self):
        results = {}
        print("computing bleu score")
        results['bleu'] = self.metric_bleu.compute()['score']
        print("computing chrf score")
        results['chrf'] = self.metric_chrf.compute()['score']
        # print("computing ter score")
        # results['ter'] = self.metric_ter.compute()['score']
        print("computing meteor score")
        results['meteor'] = self.metric_meteor.compute()['meteor']
        # print("computing bert score")
        # bertscore_res = self.metric_bertscore.compute(lang="en")  # 사용할 언어 지정
        # results['bertscore_f1'] = bertscore_res['f1'][0]
        # results['bertscore_precision'] = bertscore_res['precision'][0]
        # results['bertscore_recall'] = bertscore_res['recall'][0]

        print("metric computings all done")

        # 계산 후 초기화를 해서 다음 에폭에 대비
        # self.metric.reset()
        # AttributeError: 'Sacrebleu' object has no attribute 'reset'

        self.metric_bleu = evaluate.load("sacrebleu")
        self.metric_chrf = evaluate.load("chrf")
        # self.metric_ter = evaluate.load("ter")
        self.metric_meteor = evaluate.load("meteor")
        # self.metric_bertscore = evaluate.load("bertscore")

        return results
    
    def init_metrics_per_cat(self):
        for i in range(6):
            # 문장에 총 6개의 카테고리 존재
            self.metric_bleu_per_cat.append(evaluate.load("sacrebleu"))
            self.metric_chrf_per_cat.append(evaluate.load("chrf"))
            self.metric_meteor_per_cat.append(evaluate.load("meteor"))
            # self.metric_bertscore_per_cat.append(evaluate.load("bertscore"))

    
    def add_batch_to_metrics_per_cat(self, preds, labels, cat_list):

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        # 문장에 총 6개의 카테고리 존재
        decoded_preds_per_cat = [[] for i in range(6)]
        decoded_labels_per_cat = [[] for i in range(6)]

        assert len(decoded_preds) == len(cat_list)

        for i, cat in enumerate(cat_list):
            decoded_preds_per_cat[cat].append(decoded_preds[i])
            decoded_labels_per_cat[cat].append(decoded_labels[i])
        
        for i in range(6):
            if decoded_preds_per_cat[i]:
            # 배치에 특정 카테고리 문장이 없을 수도 있으므로 if로 확인
                self.metric_bleu_per_cat[i].add_batch(predictions=decoded_preds_per_cat[i], references=decoded_labels_per_cat[i])
                self.metric_chrf_per_cat[i].add_batch(predictions=decoded_preds_per_cat[i], references=decoded_labels_per_cat[i])
                self.metric_meteor_per_cat[i].add_batch(predictions=decoded_preds_per_cat[i], references=decoded_labels_per_cat[i])
                # self.metric_bertscore_per_cat[i].add_batch(predictions=decoded_preds_per_cat[i], references=decoded_labels_per_cat[i])


    def compute_metrics_per_cat(self):
        results = {}
        print("computing bleu score per cat")
        for i in range(6):
            results[f'bleu_{i}'] = self.metric_bleu_per_cat[i].compute()['score']

        print("computing chrf score per cat")
        for i in range(6):
            results[f'chrf_{i}'] = self.metric_chrf_per_cat[i].compute()['score']

        print("computing meteor score per cat")
        for i in range(6):
            results[f'meteor_{i}'] = self.metric_meteor_per_cat[i].compute()['meteor']

        # print("computing bert score per cat")
        # for i in range(6):
        #     bertscore_res = self.metric_bertscore_per_cat[i].compute(lang="en")  # 사용할 언어 지정
        #     results['bertscore_f1_{i}'] = bertscore_res['f1'][0]
        #     results['bertscore_precision_{i}'] = bertscore_res['precision'][0]
        #     results['bertscore_recall_{i}'] = bertscore_res['recall'][0]

        print("metric per cat computings all done")

        self.metric_bleu_per_cat = []
        self.metric_chrf_per_cat = []
        self.metric_meteor_per_cat = []
        # self.metric_bertscore_per_cat = []

        return results


    # https://huggingface.co/learn/llm-course/chapter7/4?fw=pt#metrics
    # def compute_metrics(self, preds, labels):

    #     if isinstance(preds, tuple):
    #         preds = preds[0]

    #     decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

    #     decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

    #     # Some simple post-processing
    #     decoded_preds = [pred.strip() for pred in decoded_preds]
    #     decoded_labels = [[label.strip()] for label in decoded_labels]

    #     result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)
    #     result = {"bleu": result["score"]}

    #     return result

# def convert_examples_to_features(tokenizer, max_token_length, examples):
# @@@@@@@@@ examples가 첫번쨰 인자가 아니면 
# @@@@@@@@@ partial과 dataset_dict.map의 인자 전달 방식이 충돌해서
# @@@@@@@@@ TypeError: convert_examples_to_features() got multiple values for argument 'tokenizer' 에러가 발생한다
def convert_examples_to_features(examples, tokenizer, max_token_length):
    model_inputs = tokenizer(examples['kor'],
                             text_target=examples['en'],
                             max_length=max_token_length, truncation=True)
    # 여기서 첫번째 인자와 두번째 인자의 순서를 바꾸면 한영 번역 대신 영한 번역용으로 인풋과 타겟이 토큰화된다
    return model_inputs


def collate_fn(batch, start_idx, end_idx, padding_idx, unk_idx):
    # print('Original:\n', batch)
    # print("".center(50, "-"))
    # batch는 [{'kor':..., 'en':..., 'cat':숫자, 'input_ids':[...], 'attention_mask':[1, ...], 'labels': [...]}, ...] 형태
    
    # keys = batch[0].keys()
    keys = ['kor', 'en', 'cat', 'input_ids', 'attention_mask', 'labels', 'length'] 
    # @@@ length는 input_ids의 토큰 개수
    # print(f"==>> keys: {keys}")
    # print("".center(50, "-"))
    
    # new_batch = {k:[] for k in keys}
    # new_batch['decoder_inputs'] = []
    
    # for b in batch:
    #     for k,v in b.items():
    #         if k == 'input_ids' or k == 'attention_mask':
    #             new_batch[k].append(torch.LongTensor(v))
    #         elif k == 'labels':
    #             new_batch[k].append(torch.LongTensor(v))
    #             new_batch['decoder_inputs'].append(torch.LongTensor([65001] + v[:-1]))
    #         else:
    #             new_batch[k].append(v)

    # key값별로 value 다 모으기
    new_batch = {k:[b[k] for b in batch] for k in keys}

    # list들 LongTensor로 변환
    new_batch['decoder_inputs'] = [torch.LongTensor([start_idx] + label[:-1]) for label in new_batch['labels']]
    # print(f"==>> new_batch['decoder_inputs']: {new_batch['decoder_inputs']}")
    new_batch['input_ids'] = [torch.LongTensor(inp) for inp in new_batch['input_ids']]
    new_batch['labels'] = [torch.LongTensor(label) for label in new_batch['labels']]
    new_batch['attention_mask'] = [torch.LongTensor(mask) for mask in new_batch['attention_mask']]

    new_batch['ntokens'] = sum([l.numel() for l in new_batch['labels']])

    new_batch['ntokens_input'] = sum(new_batch['length'])

    # decoder input의 attention mask 생성
    new_batch['decoder_mask'] = [torch.ones_like(d_inp, dtype=torch.long) for d_inp in new_batch['decoder_inputs']]


    # 각 input과 target 텐서를 패딩
    padded_inputs = pad_sequence(new_batch['input_ids'], batch_first=True, padding_value=padding_idx)
    new_batch['input_ids'] = padded_inputs
    padded_decoder_inputs = pad_sequence(new_batch['decoder_inputs'], batch_first=True, padding_value=padding_idx)
    new_batch['decoder_inputs'] = padded_decoder_inputs
    padded_targets = pad_sequence(new_batch['labels'], batch_first=True, padding_value=padding_idx)
    new_batch['labels'] = padded_targets

    # attention 마스크는 패딩(padding_idx) 대신 False(0)을 입력
    padded_masks = pad_sequence(new_batch['attention_mask'], batch_first=True, padding_value=0)
    new_batch['attention_mask'] = padded_masks

    padded_d_masks = pad_sequence(new_batch['decoder_mask'], batch_first=True, padding_value=0)
    new_batch['decoder_mask'] = padded_d_masks
    
    return new_batch


# 배치 안 총 토큰 개수를 일정하게 만들어주는 custom sampler
class TokenBatchSampler(Sampler):
    def __init__(self, dataset, target_tokens=2000, max_batch_samples=64, total_token_count = 40586486, mean_token_count = 32):
        # dataset에 'length' 컬럼 있어야 함
        self.lengths = dataset["length"]

        self.target_tokens = target_tokens
        self.max_batch_samples = max_batch_samples

        self.num_samples = len(dataset)
        # torch.randperm에 입력할 데이터셋 길이

        # __len__에 쓰이는 변수들
        self.total_token_count = total_token_count
        self.mean_token_count = mean_token_count
        self.len_bool = self.max_batch_samples * self.mean_token_count < self.target_tokens

        # set_epoch으로 에폭마다 증가하게 해서
        # randperm 결과 매번 다르게 하기
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        
    # 샘플러 iterator 정의 (__next__ 메소드가 불릴 때마다 yield 한번씩)
    # TODO: randperm 시드 설정 문제 및 에폭 마다 다르게 indices를 생성했을 때, worker들이 그 indices를 공유하게 하기
    def __iter__(self):
        indices = torch.randperm(self.num_samples).tolist()
        # randperm(n)은 [0, n) 사이 정수의 permutation을 반환
        # @@@ 각 epoch 마다 __iter__가 실행되어 새 iterator를 반환하므로
        # @@@ torch.randperm이 새로 실행되어 에폭마다 idx permutation은 다르다
        # @@@ @@@ 그러나 torch.manual_seed(42)로 시드가 고정된 경우 모든 에폭 동일 permutation 생성
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@ num_workers > 0 인 경우 여러 worker가 동일 인덱스를 순회하지 않도록 indices를 쪼개줘야 한다
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            # 현재 __iter__를 호출한 worker id (ex: 0, 1, 2, 3)
            num_workers = worker_info.num_workers
            # 총 worker 수 (ex: 4)
            indices = indices[worker_id::num_workers]
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        
        i = 0
        while i < len(indices): # 데이터셋의 데이터 전부를 처리할때까지 while loop
            batch = []
            current_tokens = 0
            # 배치 내 총 토큰수 카운트
            
            while (current_tokens < self.target_tokens and 
                   len(batch) < self.max_batch_samples and 
                   i < len(indices)):
            # 현재 배치 내 총 토큰 수가 target_tokens=10000 보다 작고
            # 배치 내 sample 개수가 max_batch_samples=64 보다 작고
            # 데이터셋에 남은 데이터가 있을때 while loop
                
                idx = indices[i]
                # if current_tokens + self.lengths[idx] <= self.target_tokens:
                #     batch.append(idx)
                #     current_tokens += self.lengths[idx]
                # 이렇게 두면 if 조건문이 false가 될 때의 idx를 skip 하는 문제가 있음
                if current_tokens + self.lengths[idx] >= self.target_tokens * 1.2:
                    # 갑자기 매우 큰 문장이 들어와 배치에 입력하면 
                    # self.target_tokens * 1.2 보다 커질 경우
                    # 큰 문장을 포함하기 전 배치를 바로 yield
                    break
                    # @@@ 여기서 break 했는데 현재까지의 batch 길이가 1이어서 yield 안되는 경우
                    # @@@ while i < self.num_samples:로 돌아가
                    # @@@ batch가 초기화된다 ==> 매우 큰 문장 바로 전 문장 skip 
                    # @@@ @@@ 단일 문장으로 self.target_tokens * 1.2이 되는 경우는 없다고 가정
                    # @@@ @@@ 있을 경우 무한 루프 발생

                batch.append(idx)
                current_tokens += self.lengths[idx]

                i += 1
            
            if len(batch) >= 1:
                yield batch

    # __len__: 총 배치 수를 알려주는 메소드
    # # 정확한 값이 어려우면 근사치여도 되지만 프로그레스 바가 부정확해진다
    def __len__(self):
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@ 40586486, 31.66 값은 label 기준이므로 수정 필요
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # 학습셋 총 토큰 40586486개
        # 데이터 당 평균 토큰 수 31.66개 ~ 32

        if self.len_bool:
        # # self.max_batch_samples * self.mean_token_count가 self.target_tokens 보다 작으면 
        # # 배치의 총 토큰 수가 self.target_tokens을 채우기 전에 self.max_batch_samples 도달
            return (self.num_samples // self.max_batch_samples) + 1

        # => (40586486 // 10000) + 1
        return (self.total_token_count // self.target_tokens) + 1

# Dataset.sort("length")로 bucketing 시 사용하는 custom sampler
class SortedTokenBatchSampler(Sampler):
    def __init__(self, dataset, target_tokens=2000, max_batch_samples=64, total_token_count = 40586486, mean_token_count = 32):
        self.lengths = dataset["length"]
        self.target_tokens = target_tokens
        self.max_batch_samples = max_batch_samples
        self.num_samples = len(dataset)

        # __len__에 쓰이는 변수들
        self.total_token_count = total_token_count
        self.mean_token_count = mean_token_count
        self.len_bool = self.max_batch_samples * self.mean_token_count < self.target_tokens

        # set_epoch으로 에폭 값 받아와서 짝수 에폭, 홀수 에폭 indices 방향 바꾸기
        self.epoch = 0

    # 현재 몇 epoch인지 받아오는 메소드 override
    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # sort된 순서 유지 (randperm 제거)
        # indices = list(range(self.num_samples))

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            # 현재 __iter__를 호출한 worker id (ex: 0, 1, 2, 3)
            num_workers = worker_info.num_workers
            # 총 worker 수 (ex: 4)
            indices = list(range(worker_id, self.num_samples, num_workers))
        else:
            indices = list(range(self.num_samples))


        if self.epoch % 2 == 1:
        # if self.epoch % 2 == 0: # 임시로 긴문장부터 학습해서 메모리 사용량 체크 시
            # epoch은 0부터 시작=> 0 epoch은 짧은 문장부터 학습하고
            # 1 epoch은 reverse해서 긴 문장부터 학습
            indices.reverse()
        
        i = 0
        while i < len(indices):
            batch = []
            current_tokens = 0
            
            # sort된 상태에서 토큰 수 맞추기
            while (current_tokens < self.target_tokens and 
                   len(batch) < self.max_batch_samples and 
                   i < len(indices)):
                
                idx = indices[i]

                if current_tokens + self.lengths[idx] >= self.target_tokens * 1.2:
                    break

                batch.append(idx)
                current_tokens += self.lengths[idx]

                i += 1
            
            if len(batch) >= 1:
                # print(f"==>> current_tokens: {current_tokens}")
                yield batch

    def __len__(self):
        if self.len_bool:
            return (self.num_samples // self.max_batch_samples) + 1
        
        return (self.total_token_count // self.target_tokens) + 1

