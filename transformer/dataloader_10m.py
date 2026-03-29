import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Sampler, get_worker_info
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict, ClassLabel, load_from_disk

# SacreBLEU 등 메트릭을 제공하는 huggingface의 라이브러리
import evaluate

import multiprocessing

from functools import partial
# 함수 일부 인자 고정에 사용

from collections import defaultdict
# custom sampler에서 사용

class Loaders_10M():
    def __init__(
            self,
            data_path="/home/paokimsiwoong/workspace/github.com/paokimsiwoong/transformer_experiments/transformer/data_10m.csv",
            test_path="/home/paokimsiwoong/workspace/github.com/paokimsiwoong/transformer_experiments/transformer/data_test.csv",
            max_token_length = 512,
            target_tokens = None,
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
        # Huggingface AI hub 번역 데이터셋 10개 모음
        # https://huggingface.co/datasets/nayohan/aihub-en-ko-translation-12m 
        # kor, en, style, cat 4개의 칼럼으로 구성
        # style
        # # 문어체    6522489
        # # 구어체    2913310
        # # 혼재     1260217
        # cat
        # # 과학/기술/학술자료(0), 일상/대화(1), 뉴스/시사(2), 문화/예술/역사(3), 법률/행정(4), 의학/보건(5), 특허(6), 금융/경제(7)
        # # 0    3551719
        # # 1    2913310
        # # 3    1260217
        # # 2    1181603
        # # 5     653544
        # # 4     621916
        # # 6     358307
        # # 7     155400
        # data_10m.csv와 data_10m_test.csv(기존 데이터셋과 완전히 동일한 test셋은 data_test.csv) 사용으로 변경해야함
        
        # 1-2) test 셋 로드
        testset = load_dataset("csv", data_files=test_path)['train']
        # 테스트셋은 기존의 AI hub 한국어-영어 번역(병렬) 말뭉치 데이터셋의 test 셋 부분을 그대로 사용
        # # Huggingface 데이터셋에서 이 테스트셋과 중복되는 부분은 사전에 제거한 상태
        # testset의 cat은 dataset의 cat과 분류과 다르게 되어 있다
        # testset 'cat' 열
        # 0: 구어체 8000
        # 1: 대화체 2000
        # 2: 문어체_뉴스 16028
        # 3: 문어체_한국문화 2013
        # 4: 문어체_조례 2006
        # 5: 문어체_지자체웹사이트 2002

        # 2) 카테고리 컬럼의 고유 클래스 찾아서 ClassLabel 객체 생성
        unique_classes = dataset.unique('domain')  
        # unique_classes.sort()
        print(f"==>> unique_classes: {unique_classes}")

        class_label = ClassLabel(names=unique_classes)

        # 3) 기존 컬럼 타입 변경 (캐스팅)
        dataset = dataset.cast_column('domain', class_label)

        # 4) stratify_by_column으로 분할
        train_valid = dataset.train_test_split(test_size=0.2, seed=seed, stratify_by_column='domain')

        dataset_dict = DatasetDict({
            'train': train_valid['train'],
            'validation': train_valid['test'],
        })

        # testset은 미리 정해진 것을 가져와서 쓰므로 train_test_split은 한번만
        testset_dict = DatasetDict({
            'test': testset,
        })

        NUM_CPU = multiprocessing.cpu_count()
        # print(f"==>> NUM_CPU: {NUM_CPU}")

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

        self.testsets = testset_dict.map(
            cetf,
            batched=True,
            num_proc=NUM_CPU
        )

        def add_length(batch):
            batch["length"] = [len(inp) for inp in batch["input_ids"]] 
            # @@@ map함수에 batched=True여야함
            # @@@ False이면 batch["length"] = len(batch["input_ids"])
            batch["length_label"] = [len(label) for label in batch["labels"]] 
            return batch
        
        self.datasets = self.datasets.map(add_length, batched=True, num_proc=NUM_CPU)
        self.testsets = self.testsets.map(add_length, batched=True, num_proc=NUM_CPU)

        print(f"==>> self.datasets: {self.datasets}")
        print(f"==>> self.testsets: {self.testsets}")

        self.target_tokens = target_tokens
        # loops.py에서 각 에폭 시작 시점에 self.target_tokens 확인 후 
        # None이 아니면 self.sampler의 set_epoch_indices 메소드 실행

        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test

        if target_tokens is not None:
            self.sampler = TokenPadBatchSampler(
                self.datasets["train"], 
                target_tokens=target_tokens,
                max_batch_samples=batch_size_train,
            )

        self.train_set = self.datasets['train']
        self.val_set = self.datasets['validation']
        self.test_set = self.testsets['test']

        c_fn = partial(collate_fn, start_idx=self.start_idx, end_idx=self.end_idx, padding_idx=self.padding_idx, unk_idx=self.unk_idx)
        c_fn_test = partial(collate_fn_test, start_idx=self.start_idx, end_idx=self.end_idx, padding_idx=self.padding_idx, unk_idx=self.unk_idx)

        if target_tokens is None:
            self.loader_train = DataLoader(self.train_set, batch_size=batch_size_train, collate_fn=c_fn, shuffle=True, num_workers=num_workers, pin_memory=True)
        else:
            # @@@ 배치 별 총 토큰 수 일정하게 유지하기 위해 batch_size 대신 batch_sampler 사용
            self.loader_train = DataLoader(self.train_set, batch_sampler=self.sampler, collate_fn=c_fn, shuffle=False, num_workers=num_workers, pin_memory=True)
        # 학습시에만 shuffle=True
        self.loader_val = DataLoader(self.val_set, batch_size=batch_size_val, collate_fn=c_fn, shuffle=False, num_workers=val_num_workers, pin_memory=True)
        self.loader_test = DataLoader(self.test_set, batch_size=batch_size_test, collate_fn=c_fn_test, shuffle=False, num_workers=val_num_workers, pin_memory=True)

        # 테스트 루프에 쓰일 메트릭 초기화
        self.metric_bleu = evaluate.load("sacrebleu")
        self.metric_chrf = evaluate.load("chrf")
        self.metric_meteor = evaluate.load("meteor")

        self.metric_bleu_per_cat = []
        self.metric_chrf_per_cat = []
        self.metric_meteor_per_cat = []

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
        self.metric_meteor.add_batch(predictions=decoded_preds, references=decoded_labels)

    def compute_metrics(self):
        results = {}
        print("computing bleu score")
        results['bleu'] = self.metric_bleu.compute()['score']
        print("computing chrf score")
        results['chrf'] = self.metric_chrf.compute()['score']
        print("computing meteor score")
        results['meteor'] = self.metric_meteor.compute()['meteor']

        print("metric computings all done")

        self.metric_bleu = evaluate.load("sacrebleu")
        self.metric_chrf = evaluate.load("chrf")
        self.metric_meteor = evaluate.load("meteor")

        return results
    
    def init_metrics_per_cat(self):
        for i in range(6):
            # 문장에 총 6개의 카테고리 존재
            self.metric_bleu_per_cat.append(evaluate.load("sacrebleu"))
            self.metric_chrf_per_cat.append(evaluate.load("chrf"))
            self.metric_meteor_per_cat.append(evaluate.load("meteor"))

    
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

        print("metric per cat computings all done")

        self.metric_bleu_per_cat = []
        self.metric_chrf_per_cat = []
        self.metric_meteor_per_cat = []

        return results

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
    keys = ['kor', 'en', 'domain', 'style', 'input_ids', 'attention_mask', 'labels', 'length', 'length_label']
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

    new_batch['ntokens'] = sum(new_batch['length_label'])

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

# collate_fn과 동일하지만 testset에 맞게 style key 제거
def collate_fn_test(batch, start_idx, end_idx, padding_idx, unk_idx):
    # print('Original:\n', batch)
    # print("".center(50, "-"))
    # batch는 [{'kor':..., 'en':..., 'cat':숫자, 'input_ids':[...], 'attention_mask':[1, ...], 'labels': [...]}, ...] 형태
    
    # keys = batch[0].keys()
    keys = ['kor', 'en', 'cat', 'input_ids', 'attention_mask', 'labels', 'length', 'length_label']
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

    new_batch['ntokens'] = sum(new_batch['length_label'])

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
    def __init__(self, dataset, target_tokens=2000, max_batch_samples=64, total_token_count = 268075627, mean_token_count = 23):
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

        self.indices = []

    def set_epoch_indices(self):
    # 각 에폭 시작 시점에 한번씩 호출 필요
        self.indices = torch.randperm(self.num_samples).tolist()
        
    # 샘플러 iterator 정의 (__next__ 메소드가 불릴 때마다 yield 한번씩)
    def __iter__(self):
        indices = self.indices

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@ num_workers > 0 인 경우 여러 worker가 동일 인덱스를 순회하지 않도록 indices를 쪼개줘야 한다
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            # 현재 __iter__를 호출한 worker id (ex: 0, 1, 2, 3)
            num_workers = worker_info.num_workers
            # 총 worker 수 (ex: 4)
            indices = self.indices[worker_id::num_workers]
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        
        i = 0
        while i < len(indices): # 데이터셋의 데이터 전부를 처리할때까지 while loop
            batch = []
            current_tokens = 0
            # 배치 내 총 토큰수 카운트
            
            while (current_tokens < self.target_tokens and 
                #    len(batch) < self.max_batch_samples and 
                   i < len(indices)):
            # 현재 배치 내 총 토큰 수가 target_tokens 보다 작고
            # 배치 내 sample 개수가 max_batch_samples 보다 작고
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
        # 학습셋 총 토큰 29704824개
        # 데이터 당 평균 토큰 수 23.17개 ~ 23
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@ 40586486, 31.66 값은 label 기준
        # 학습셋 총 토큰 40586486개
        # 데이터 당 평균 토큰 수 31.66개 ~ 32
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        # if self.len_bool:
        # # self.max_batch_samples * self.mean_token_count가 self.target_tokens 보다 작으면 
        # # 배치의 총 토큰 수가 self.target_tokens을 채우기 전에 self.max_batch_samples 도달
            # return (self.num_samples // self.max_batch_samples) + 1

        # => (40586486 // 10000) + 1
        return (self.total_token_count // self.target_tokens) + 1


# 각 배치 내 최대길이 * 배치 갯수 값을 일정값으로 고정하는 sampler
class TokenPadBatchSampler(TokenBatchSampler):
    def __init__(self, dataset, target_tokens=2000, max_batch_samples=64, total_token_count = 268075627, mean_token_count = 23):
        super().__init__(dataset, target_tokens, max_batch_samples, total_token_count, mean_token_count)
        
    # 샘플러 iterator 정의 (__next__ 메소드가 불릴 때마다 yield 한번씩)
    def __iter__(self):
        indices = self.indices

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@ num_workers > 0 인 경우 여러 worker가 동일 인덱스를 순회하지 않도록 indices를 쪼개줘야 한다
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            # 현재 __iter__를 호출한 worker id (ex: 0, 1, 2, 3)
            num_workers = worker_info.num_workers
            # 총 worker 수 (ex: 4)
            indices = self.indices[worker_id::num_workers]
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        
        i = 0
        while i < len(indices): # 데이터셋의 데이터 전부를 처리할때까지 while loop
            batch = []
            current_tokens = 0
            # 배치 내 총 토큰수 카운트

            b_max_length = 0
            # 배치 내 문장 최대 길이
            current_tokenpads = 0
            # 배치 내 총 토큰+패드 수 카운트
            
            while (current_tokens < self.target_tokens and 
                   len(batch) * b_max_length < self.target_tokens and
                   i < len(indices)):
            # 현재 배치 내 총 토큰 수가 target_tokens 보다 작고
            # 배치 내 sample 개수 * 배치 내 문장 토큰 최대 길이가 target_tokens 보다 작고
            # 데이터셋에 남은 데이터가 있을때 while loop
                
                idx = indices[i]

                if self.lengths[idx] <= b_max_length:
                    # 새 문장 토큰 길이가 기존 최대값보다 작거나 같으면
                    if current_tokenpads + b_max_length > self.target_tokens:
                        # 최대값 * (배치 개수 + 1(새문장)) 값이 self.target_tokens보다 큰지 확인하고
                        # 클 경우 새문장을 제외한 기존 배치만 yield
                        break
                else:
                    # 새 문장 토큰 길이가 기존 최대값보다 크면
                    if (len(batch) + 1) * self.lengths[idx] > self.target_tokens:
                        # 새문장 토큰 길이를 최대값 기준으로 배치 내 총 토큰+패드 개수를 계산 했을 때
                        # 이 값이 self.target_tokens 보다 크면 갱신을 취소하고
                        # 새 문장을 제외한 기존 배치만 yield
                        break

                    b_max_length = self.lengths[idx]
                    # 최대값을 갱신해도 총 토큰+패드 개수가 self.target_tokens를 안넘으면
                    # 최대값을 갱신

                if current_tokens + self.lengths[idx] > self.target_tokens:
                    # 이번 문장이 들어오면 배치 내 총 토큰 수가
                    # self.target_tokens 보다 커질 경우
                    # 큰 문장을 포함하기 전 배치를 바로 yield
                    break

                batch.append(idx)
                current_tokens += self.lengths[idx]
                current_tokenpads = len(batch) * b_max_length

                i += 1
            
            if len(batch) >= 1:
                yield batch

    # __len__: 총 배치 수를 알려주는 메소드
    # # 정확한 값이 어려우면 근사치여도 되지만 프로그레스 바가 부정확해진다
    def __len__(self):
        # 토큰 개수 268075627
        # label 토큰 개수 329348725
        # 토큰+패드 기준으로 배치 크기를 제한하므로
        # __len__값은 부정확
        return (self.total_token_count // self.target_tokens) + 1