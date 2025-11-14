import os
import os.path as osp
import random
from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from tqdm import tqdm


from dataloader import Loaders
from models import Transformer
from loss import LabelSmoothing


def parse_args():
    parser = ArgumentParser()

    # Conventional args

    # 학습 데이터 경로
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get(
            "SM_DATA_DIR",
            "/home/paokimsiwoong/workspace/github.com/paokimsiwoong/transformer_experiments/transformer/data.csv",
        ),
    )

    # pth 파일 저장 경로
    parser.add_argument(
        "--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/home/paokimsiwoong/workspace/github.com/paokimsiwoong/transformer_experiments/transformer/pths")
    )

    # resume 파일 이름
    parser.add_argument("--resume_name", type=str, default="")

    # random seed
    parser.add_argument("--seed", type=int, default=42)

    # debug 여부
    parser.add_argument('--debug', action='store_true', default=False, help='enable debug mode')

    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num_workers", type=int, default=4)
    # gpu 개수 * 4로 설정하는 것이 좋다는 의견이 있음

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--val_num_workers", type=int, default=4)

    parser.add_argument("--max_token_length", type=int, default=512)
    parser.add_argument("--q_dim", type=int, default=512)


    parser.add_argument("--len_vocab", type=int, default=64101)
    parser.add_argument("--start_idx", type=int, default=64100)
    parser.add_argument("--end_idx", type=int, default=1)
    parser.add_argument("--padding_idx", type=int, default=0)
    parser.add_argument("--unk_idx", type=int, default=2)


    parser.add_argument("--label_smoothing", type=float, default=0.1)


    parser.add_argument("--learning_rate", type=float, default=1)
    # parser.add_argument("--weight_decay", type=float, default=0.00005)
    # parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--warmup_steps", type=int, default=16000)
    # @@@ 전체 step의 5~10% (128만 문장을 batch_size 8 => 160000 스텝 => 10%는 16000)
    parser.add_argument("--max_epoch", type=int, default=10)

    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--val_interval", type=int, default=1)

    # ????

    # parser.add_argument("--mp", action="store_false")
    # https://stackoverflow.com/questions/60999816/argparse-not-parsing-boolean-arguments
    # mixed precision 사용할 지 여부

    # parser.add_argument("--wandb_mode", type=str, default="online")
    # parser.add_argument("--wandb_mode", type=str, default="offline")
    parser.add_argument("--wandb_mode", type=str, default="disabled")
    # wandb mode
    parser.add_argument("--wandb_run_name", type=str, default="KorEnTransformer")
    # wandb run name

    args = parser.parse_args()

    return args


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# LambdaLR 스케쥴러에 사용하는 함수
def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )



def train(
    data_dir,
    model_dir,
    device,
    num_workers,
    batch_size,
    val_num_workers,
    val_batch_size,
    max_token_length,
    q_dim,
    len_vocab,
    start_idx,
    end_idx,
    padding_idx,
    unk_idx,
    label_smoothing,
    learning_rate,
    warmup_steps,
    max_epoch,
    val_interval,
    save_interval,
    resume_name,
    seed,
    # mp,
    wandb_mode,
    wandb_run_name,
    debug,
):
    print("".center(50, "-"))
    print("".center(50, "-"))
    print("".center(50, "-"))

    time_start = datetime.now()

    train_start = time_start.strftime("%Y%m%d_%H%M%S")

    set_seed(seed)

    if not osp.exists(model_dir):
        os.makedirs(model_dir)

    # batch_size = batch_size

    # val_batch_size = val_batch_size

    # -- early stopping flag
    # patience = patience
    # counter = 0

    # 데이터셋
    loaders = Loaders(data_path=data_dir, max_token_length=max_token_length, batch_size_train=batch_size, num_workers=num_workers, batch_size_val=val_batch_size, batch_size_test=val_batch_size, val_num_workers=val_num_workers, start_idx=start_idx, end_idx=end_idx, padding_idx=padding_idx, unk_idx=unk_idx, seed=seed)

    data_load_end = datetime.now()
    data_load_time = data_load_end - time_start
    data_load_time = str(data_load_time).split(".")[0]
    print(f"==>> data_load_time: {data_load_time}")

    print("".center(50, "-"))

    # Initialize the model
    model = Transformer(src_len_vocab=len_vocab, tgt_len_vocab=len_vocab, start_idx=start_idx, end_idx=end_idx, padding_idx=padding_idx, unk_idx=unk_idx, q_dim=q_dim, k_dim=q_dim, v_dim=q_dim, h_dim=(q_dim * 4))
    # KETI-AIR/ke-t5-base tokenizer의 한영 통합 토큰 종류 수는 64100 + 시작 토큰 1개 추가해서 = 64101개

    # 파라메터 초기화 임시 위치
    for p in model.parameters():
        if p.dim() > 1:
            # nn.init.xavier_uniform(p)
            nn.init.xavier_uniform_(p)

    load_dict = None

    if resume_name:
        load_dict = torch.load(
            osp.join(model_dir, f"{resume_name}.pth"), map_location="cpu"
        )
        model.load_state_dict(load_dict["model_state_dict"])

    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        # lr=learning_rate,
        lr=1,
        # betas=(0.9, 0.999),
        # annotated transformer 값 따라하기
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda step: rate(step, model_size=q_dim, factor=1, warmup=warmup_steps))

    if resume_name:
        optimizer.load_state_dict(load_dict["optimizer_state_dict"])
        scheduler.load_state_dict(load_dict["scheduler_state_dict"])
    #     scaler.load_state_dict(load_dict["scaler_state_dict"])

    criterion = LabelSmoothing(size=len_vocab, padding_idx=padding_idx, smoothing=label_smoothing)

    print(f"Start training..")

    print("".center(50, "-"))
    print("".center(50, "-"))
    print("".center(50, "-"))

    wandb.init(
        project="transformer",
        entity="pao-kim-si-woong",
        config={
            "lr": learning_rate,
            "dataset": "AIHub 한국어-영어 번역(병렬) 말뭉치",
            "n_epochs": max_epoch,
            "loss": "Label Smoothing",
            "notes": "transformer 한영 번역 실험",
        },
        name=wandb_run_name + "_" + train_start,
        mode=wandb_mode,
    )

    wandb.watch((model,))

    # best_loss = np.inf
    # best_auc = 0
    val_bleu = 0
    best_bleu = 0
    # best_perplexity = 0

    num_batches_train = len(loaders.loader_train)

    for epoch in range(max_epoch):
        model.train()

        epoch_start = datetime.now()

        epoch_loss = 0
        epoch_total_tokens = 0
        

        for step, batch in tqdm(
            enumerate(loaders.loader_train),
            total=num_batches_train,
        ):
            # batch는 'kor', 'en', 'cat', 'input_ids', 'attention_mask', 'decoder_inputs', 'decoder_mask', 'labels', 'ntokens'들을 키로 가지는 dict

            # if step == 10:
            #     break

            inputs = batch['input_ids'].to(device)
            # (batch_size, src_seq_len)

            # teacher forcing에 사용할 gts
            gts = batch['decoder_inputs'].to(device)
            # (batch_size, tgt_seq_len)

            # loss 계산에 사용할 정답 레이블
            labels = batch['labels'].to(device)
            # (batch_size, tgt_seq_len)

            # 이번 배치의 정답 토큰 총 개수
            # batch_ntokens = labels.numel()
            # b * tgt_seq_len
            # TODO: 패드 토큰 개수 빼야 하는지 확인 필요
            # @@@ annotated transformer는 self.ntokens = (self.tgt_y != pad).data.sum()로 패드 토큰 개수 제외함
            batch_ntokens = batch['ntokens']
            # collate_fn에서 계산하는 방식
            # @@@ 단순히  batch['decoder_mask'].sum()해도 동일한 값이 나온다(실제 토큰부분 1, 패딩 부분 0으로 되어 있으므로)

            epoch_total_tokens += batch_ntokens


            x_masks = batch['attention_mask'].to(device)
            gt_masks = batch['decoder_mask'].to(device)

            optimizer.zero_grad()

            out = model(inputs, gts, x_masks, gt_masks)
            # out.size는 (b, tgt_seq_len, tgt_len_vocab)


            # loss 계산 시에는 out은 (b * tgt_seq_len, tgt_len_vocab)
            # labels는 (b * tgt_seq_len)로 변경 후 입력
            # loss = criterion(out.view(-1, out.size(-1)), labels.view(-1))
            loss = criterion(out.contiguous().view(-1, out.size(-1)), labels.contiguous().view(-1))
            # loss는 모든 토큰 loss값을 더한 값
            # annotated transformer는 backward 하기 전에 batch_ntokens로 나누고 backward
            # 로그되는 값은 나누기 전 모든 토큰 loss 총합 그대로
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # perplexity 답변:
            # 배치의 loss를 label 토큰 총 개수(패딩 제외)로 나누고 backward를 하는 것은, 
            # loss를 토큰 단위로 정규화(normalize) 하기 위함입니다. 
            # 이렇게 하면 각 배치의 loss가 토큰 수에 따라 달라지지 않도록 하여, 학습이 더 안정적으로 진행됩니다.
                # 배치마다 토큰의 개수가 다름
                # ==> loss를 배치 토큰 개수로 나누지 않을 경우 토큰이 많은 배치(긴 문장이 많이 들어간 배치)의 그래디언트가 크게 됨
                # ==> 짧은 문장보다 긴 문장이 많을 때 학습이 잘되므로 문장 길이마다 학습 정도가 불균형하게 됨
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

            normalized_loss = loss / batch_ntokens

            normalized_loss.backward()

            loss_value = loss.item()
            normalized_loss_value = normalized_loss.item()

            # with torch.no_grad():
            # @@@ 텐서.item()은 그래프에서 분리된 순수한 숫자(float)이므로 그래디언트 계산과 무관
            wandb_step_dict = {
                "step_total_loss": loss_value,
                "step_token_loss": normalized_loss_value,
                "learning_rate": scheduler.get_last_lr()[0],
            }

            wandb.log(wandb_step_dict)

            optimizer.step()
            scheduler.step()

            
            # with torch.no_grad():
            # @@@ loss.item()은 그래프에서 분리된 순수한 숫자(float)이므로 그래디언트 계산과 무관
            epoch_loss += loss_value


        # 배치당 loss 값의 평균 계산
        epoch_mean_batch_loss = epoch_loss / num_batches_train
        # 토큰 한개당 loss 값의 평균 계산
        epoch_mean_token_loss = epoch_loss / epoch_total_tokens
  

        train_end = datetime.now()
        train_time = train_end - epoch_start
        train_time = str(train_time).split(".")[0]
        print("".center(50, "-"))
        print("".center(50, "-"))
        print(
            f"==>> epoch {epoch+1} train_time: {train_time}\nepoch_total_tokens: {epoch_total_tokens}"
        )
        print("".center(50, "-"))
        print(
            f"mean_batch_loss: {round(epoch_mean_batch_loss,4)}\n mean_token_loss: {round(epoch_mean_token_loss,4)}"
        )

        if (epoch + 1) % save_interval == 0:
            
            

            ckpt_fpath = osp.join(model_dir, f"transformer_koren_{train_start}_latest.pth")

            states = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),  # 모델의 state_dict 저장
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                # "scaler_state_dict": scaler.state_dict(),
            }

            torch.save(states, ckpt_fpath)

        # validation 주기에 따라 loss 또는 평가메트릭을 계산하고 best model을 저장
        if (epoch + 1) % val_interval == 0:

            # val 또는 test 과정에 원문, 번역문 정답, 번역문 예측 결과 저장하기

            print(f"Start validation #{epoch+1:2d}")
            model.eval()

            with torch.no_grad():
                # val_loss = 0
                # @@@ 예측 문장의 길이와 정답 문장의 길이가 다를 수 있어 LabelSmoothing으로 loss 계산 어려움

                # val_bleu = 0
                # val_perplexity = 0

                val_total_tokens = 0

                num_batches_val = len(loaders.loader_val)

                for step, batch_val in tqdm(
                    enumerate(loaders.loader_val), total=num_batches_val
                ):  
                    # if step == 10:
                    #     break

                    inputs = batch_val['input_ids'].to(device)
                    # (batch_size, src_seq_len)

                    # teacher forcing에 사용할 gts
                    # @@@ val 과정에서는 필요 없음
                    # gts = batch_val['decoder_inputs'].to(device)
                    # (batch_size, tgt_seq_len)

                    # loss 계산에 사용할 정답 레이블
                    labels = batch_val['labels'].to(device)
                    # (batch_size, tgt_seq_len)

                    batch_val_ntokens = batch_val['ntokens']

                    val_total_tokens += batch_val_ntokens

                    x_masks = batch_val['attention_mask'].to(device)
                    # @@@ val 과정에서는 필요 없음
                    # gt_masks = batch_val['decoder_mask'].to(device)

                    preds = model.inference(inputs, x_masks, labels.size(-1) * 2)
                    # (batch_size, pred_seq_len)

                    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    # result = loaders.compute_metrics(preds, labels)

                    # val_bleu += result['bleu']
                    # bleu는 배치별로 계산하지 않고 전체 예측 문장과 전체 정답 문장을 모아서 한꺼번에 BLEU를 계산하는 방식을 사용
                    # 이렇게 해야 문맥과 길이 등이 고려된 전체적인 BLEU 점수를 정확하게 측정할 수 있다

                    loaders.add_batch_to_metrics(preds, labels)
                
                result = loaders.compute_metrics()

                val_bleu = result['bleu']

            if best_bleu < val_bleu:
                print(
                    f"Best bleu performance at epoch: {epoch + 1}, {best_bleu:.4f} -> {val_bleu:.4f}"
                )
                print(f"Save model in {model_dir}")
                states = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),  # 모델의 state_dict 저장
                }

                best_ckpt_fpath = osp.join(
                    model_dir, f"transformer_koren_{train_start}_best_bleu.pth"
                )
                torch.save(states, best_ckpt_fpath)
                best_bleu = val_bleu
            #     counter = 0
            # else:
            #     counter += 1

        wandb_epoch_dict = {
            "train_batch_loss": epoch_mean_batch_loss,
            "train_token_loss": epoch_mean_token_loss,
            # "valid_loss": val_mean_loss,
            "val_bleu": val_bleu,
            # "learning_rate": scheduler.get_last_lr()[0],
        }

        wandb.log(wandb_epoch_dict)

        # scheduler.step()

        epoch_end = datetime.now()
        epoch_time = epoch_end - epoch_start
        epoch_time = str(epoch_time).split(".")[0]
        print("".center(50, "-"))
        print("".center(50, "-"))
        print(
            f"==>> epoch {epoch+1} time: {epoch_time}\nval_total_tokens: {val_total_tokens}"
        )
        print("".center(50, "-"))
        print(f"val_bleu: {val_bleu}")

        # if counter > patience:
        #     print("Early Stopping...")
        #     break

    print("".center(50, "-"))
    print("".center(50, "-"))

    # val 또는 test 과정에 원문, 번역문 정답, 번역문 예측 결과 저장하기

    print(f"Start Test")
    test_start = datetime.now()

    test_bleu = 0

    model.eval()

    with torch.no_grad():
        # test_loss = 0
        # @@@ 예측 문장의 길이와 정답 문장의 길이가 다를 수 있어 LabelSmoothing으로 loss 계산 어려움

        # test_bleu = 0
        # test_perplexity = 0

        test_total_tokens = 0

        num_batches_test = len(loaders.loader_test)

        for step, batch_test in tqdm(
            enumerate(loaders.loader_test), total=num_batches_test
        ):
            # if step == 10:
            #     break

            inputs = batch_test['input_ids'].to(device)
            # (batch_size, src_seq_len)

            # teacher forcing에 사용할 gts
            # @@@ test 과정에서는 필요 없음
            # gts = batch_test['decoder_inputs'].to(device)
            # (batch_size, tgt_seq_len)

            # loss 계산에 사용할 정답 레이블
            labels = batch_test['labels'].to(device)
            # (batch_size, tgt_seq_len)

            batch_test_ntokens = batch_test['ntokens']

            test_total_tokens += batch_test_ntokens

            x_masks = batch_test['attention_mask'].to(device)
            # @@@ test 과정에서는 필요 없음
            # gt_masks = batch_test['decoder_mask'].to(device)

            preds = model.inference(inputs, x_masks, labels.size(-1) * 2)
            # (batch_size, pred_seq_len)

            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # result = loaders.compute_metrics(preds, labels)

            # val_bleu += result['bleu']
            # bleu는 배치별로 계산하지 않고 전체 예측 문장과 전체 정답 문장을 모아서 한꺼번에 BLEU를 계산하는 방식을 사용
            # 이렇게 해야 문맥과 길이 등이 고려된 전체적인 BLEU 점수를 정확하게 측정할 수 있다

            loaders.add_batch_to_metrics(preds, labels)
        
        result = loaders.compute_metrics()

        test_bleu = result['bleu']


        test_end = datetime.now()
        test_time = test_end - test_start
        test_time = str(test_time).split(".")[0]
        print("".center(50, "-"))
        print("".center(50, "-"))
        print(
            f"==>> test time: {test_time}\ntest_total_tokens: {test_total_tokens}"
        )
        print("".center(50, "-"))
        print(f"test_bleu: {test_bleu}")

    wandb_test_dict = {
        # "test_loss": test_mean_loss,
        "test_bleu": test_bleu,
    }

    wandb.log(wandb_test_dict)

    print("".center(50, "-"))
    print("".center(50, "-"))

    time_end = datetime.now()
    total_time = time_end - time_start
    total_time = str(total_time).split(".")[0]
    print(f"==>> total time: {total_time}")

    print("".center(50, "-"))
    print("".center(50, "-"))
    print("".center(50, "-"))


def main(args):
    # debug를 위해 CUDA 호출을 CPU 코드와 동기화
    if args.__dict__['debug']:
        print("debug mode")
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # else:
    #     print("not debug")

    train(**args.__dict__)


if __name__ == "__main__":
    args = parse_args()

    main(args)