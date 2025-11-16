import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict, ClassLabel

# SacreBLEU 등 메트릭을 제공하는 huggingface의 라이브러리
import evaluate

import multiprocessing

from functools import partial
# 함수 일부 인자 고정에 사용

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
        self.tokenizer = AutoTokenizer.from_pretrained("KETI-AIR/ke-t5-base")

        special_tokens_dict = {'bos_token': '<s>'}
        self.tokenizer.add_special_tokens(special_tokens_dict)

        print(f"==>> self.tokenizer.model_max_length: {self.tokenizer.model_max_length}")

        print(f"==>> len(self.tokenizer): {len(self.tokenizer)}")

        self.max_token_length = min(max_token_length, self.tokenizer.model_max_length)

        # partial을 이용해 tokenizer, max_token_length 인자 고정
        cetf = partial(convert_examples_to_features, tokenizer=self.tokenizer, max_token_length=self.max_token_length)
        # @@@ partial과 dataset_dict.map의 인자 전달 방식이 충돌해서 
        # @@@ TypeError: convert_examples_to_features() got multiple values for argument 'tokenizer' 에러 발생

        self.datasets = dataset_dict.map(
                                cetf,
                                # lambda examples: convert_examples_to_features(examples, tokenizer=self.tokenizer, max_token_length=self.max_token_length),
                                batched=True,
                                # 이걸 쓰지 않으면 원 데이터 'en', 'kor', 'cat'가 남아서
                                # 아래서 콜레이터가 패딩을 못해서 에러남
                                # remove_columns=dataset_dict["train"].column_names,
                                num_proc=NUM_CPU)

        print(f"==>> self.datasets: {self.datasets}")

        self.train_set = self.datasets['train']
        self.val_set = self.datasets['validation']
        self.test_set = self.datasets['test']

        c_fn = partial(collate_fn, start_idx=self.start_idx, end_idx=self.end_idx, padding_idx=self.padding_idx, unk_idx=self.unk_idx)

        self.loader_train = DataLoader(self.train_set, batch_size=batch_size_train, collate_fn=c_fn, shuffle=True, num_workers=num_workers)
        # 학습시에만 shuffle=True
        self.loader_val = DataLoader(self.val_set, batch_size=batch_size_val, collate_fn=c_fn, shuffle=False, num_workers=val_num_workers)
        self.loader_test = DataLoader(self.test_set, batch_size=batch_size_test, collate_fn=c_fn, shuffle=False, num_workers=val_num_workers)

        # BLEU 계산에 필요
        self.metric_bleu = evaluate.load("sacrebleu")
        # METEOR - 단어 정렬, 동의어, 어근 등을 반영하여 인간 평가에 근접한 의미적 평가
        self.metric_meteor = evaluate.load("meteor")
        # BERTScore - 사전학습된 BERT 임베딩 기반으로 문장 수준 의미적 유사성을 평가
        self.metric_bertscore = evaluate.load("bertscore")
        # ChrF - 문자 단위 n-그램 기반으로, 한국어 같은 교착어 처리에 강하고 어휘 미스매치에 강건
        self.metric_chrf = evaluate.load("chrf")
        # TER (Translation Edit Rate) - 번역과 정답 간 편집 거리 기반 메트릭으로, 오류율을 직관적으로 파악할 수 있어 BLEU 보완에 유용
        self.metric_ter = evaluate.load("ter")

    def add_batch_to_metrics(self, preds, labels):

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        self.metric_bleu.add_batch(predictions=decoded_preds, references=decoded_labels)
        self.metric_meteor.add_batch(predictions=decoded_preds, references=decoded_labels)
        self.metric_bertscore.add_batch(predictions=decoded_preds, references=decoded_labels)
        self.metric_chrf.add_batch(predictions=decoded_preds, references=decoded_labels)
        self.metric_ter.add_batch(predictions=decoded_preds, references=decoded_labels)

    def compute_metrics(self):
        results = {}
        print("computing bleu score")
        results['bleu'] = self.metric_bleu.compute()['score']
        print("computing meteor score")
        results['meteor'] = self.metric_meteor.compute()['meteor']
        print("computing chrf score")
        results['chrf'] = self.metric_chrf.compute()['score']
        print("computing ter score")
        results['ter'] = self.metric_ter.compute()['score']  # 참고로 TER는 wer 키를 쓸 수도 있음
        print("computing bert score")
        bertscore_res = self.metric_bertscore.compute(lang="en")  # 사용할 언어 지정
        results['bertscore_f1'] = bertscore_res['f1'][0]
        results['bertscore_precision'] = bertscore_res['precision'][0]
        results['bertscore_recall'] = bertscore_res['recall'][0]

        print("metric computings all done")

        # 계산 후 초기화를 해서 다음 에폭에 대비
        # self.metric.reset()
        # AttributeError: 'Sacrebleu' object has no attribute 'reset'

        self.metric_bleu = evaluate.load("sacrebleu")
        self.metric_meteor = evaluate.load("meteor")
        self.metric_bertscore = evaluate.load("bertscore")
        self.metric_chrf = evaluate.load("chrf")
        self.metric_ter = evaluate.load("ter")

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
    keys = ['kor', 'en', 'cat', 'input_ids', 'attention_mask', 'labels']
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