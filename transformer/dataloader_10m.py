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

class Loaders():
    def __init__(
            self,
            data_path="/home/paokimsiwoong/workspace/github.com/paokimsiwoong/ml_practice/transformer/data.csv",
            max_token_length = 512,
            batch_size_train = 8,
            num_workers = 4,
            batch_size_val = 4,
            batch_size_test = 4,
            val_num_workers = 4,
            start_idx = 64100, 
            end_idx = 1, 
            padding_idx = 0, 
            unk_idx = 2,
            seed = 42,
    ):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.padding_idx = padding_idx
        self.unk_idx = unk_idx

        # 1) 데이터셋 로드
        dataset = load_from_disk(data_path)['train']

        # 4) stratify_by_column으로 분할
        # train_validtest = dataset.train_test_split(test_size=0.2, seed=seed, stratify_by_column='cat')
        # valid_test = train_validtest['test'].train_test_split(test_size=0.1, seed=seed, stratify_by_column='cat')
        train_validtest = dataset.train_test_split(test_size=0.2, seed=seed, stratify_by_column='style_class')
        valid_test = train_validtest['test'].train_test_split(test_size=0.1, seed=seed, stratify_by_column='style_class')

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
        self.tokenizer = AutoTokenizer.from_pretrained("KETI-AIR/ke-t5-base")

        special_tokens_dict = {'bos_token': '<s>'}
        self.tokenizer.add_special_tokens(special_tokens_dict)

        print(self.tokenizer.all_special_ids)
        print(self.tokenizer.all_special_tokens)

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

        print(f"==>> self.datasets: {self.datasets}")

        self.train_set = self.datasets['train']
        self.val_set = self.datasets['validation']
        self.test_set = self.datasets['test']

        c_fn = partial(collate_fn, start_idx=self.start_idx, end_idx=self.end_idx, padding_idx=self.padding_idx, unk_idx=self.unk_idx)

        self.loader_train = DataLoader(self.train_set, batch_size=batch_size_train, collate_fn=c_fn, shuffle=True, num_workers=num_workers, pin_memory=True)
        # 학습시에만 shuffle=True
        self.loader_val = DataLoader(self.val_set, batch_size=batch_size_val, collate_fn=c_fn, shuffle=False, num_workers=val_num_workers, pin_memory=True)
        self.loader_test = DataLoader(self.test_set, batch_size=batch_size_test, collate_fn=c_fn, shuffle=False, num_workers=val_num_workers, pin_memory=True)

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
    keys = ['kor', 'en', 'domain', 'cat', 'style', 'style_class', 'input_ids', 'attention_mask', 'labels']
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