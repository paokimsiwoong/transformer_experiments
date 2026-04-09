import os
import os.path as osp

from datetime import datetime

import torch
import torch.nn as nn
import wandb

import numpy as np


from arg_parser import parse_args

from dataloader import Loaders
from dataloader_10m import Loaders_10M
from dataloader_9m import Loaders_9M
from models import Transformer
from loss import LabelSmoothing
from loops import train_loop, val_loop, test_loop, train_loop_with_mp, val_loop_with_mp, test_loop_with_mp
from hooks import nan_hook, clip_grad_hook, clip_grad_embed_hook
from utils import set_seed, check_nan_in_parameters, WeightTying

import gc
# gc.collect()로 python 객체 정리

def train(
    args_dicts, # unpack하지 않은 dict도 받아서 pth 안에 같이 저장하기
    data_dir,
    model_dir,
    image_dir,
    device,
    num_workers,
    target_tokens,
    batch_size,
    val_num_workers,
    val_batch_size,
    test_batch_size,
    max_token_length,
    q_dim,
    weight_tying,
    # decouple_src_tgt_embed,
    # decouple_ffc_tgt_embed,
    # decouple_embed_ffc,
    # len_vocab,
    # start_idx,
    # end_idx,
    # padding_idx,
    # unk_idx,
    tokenizer,
    label_smoothing,
    learning_rate,
    learning_factor,
    warmup_steps,
    adjusted,
    # total_steps,
    # decay_rate,
    max_norm,
    max_epoch,
    save_interval,
    val_interval,
    test_interval,
    resume_name,
    seed,
    mp,
    accum_steps,
    wandb_mode,
    wandb_run_name,
    viz,
    grad_cp,
    debug,
    train_break,
    val_break,
    test_break,
):
    print("".center(50, "-"))
    print("".center(50, "-"))
    print("".center(50, "-"))

    time_start = datetime.now()

    train_start = time_start.strftime("%Y%m%d_%H%M%S")

    set_seed(seed)

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # @@@ 공유 메모리 사용 문제 해결
    # os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:512"
    # os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
    # @@@ 사용하면 메모리 사용량이 들쭉날쭉 계속 변하면서 느려짐
    # max_split_size_mb:512로 큰 텐서 할당시 블록 개수를 줄여 조각화 방지 
    # expandable_segments:True는 기존 블록의 크기가 부족하면 동적으로 크기를 늘려서 새 블록 할당을 최소화

    # PYTORCH_CUDA_ALLOC_CONF는 deprecated
    # os.environ["PYTORCH_ALLOC_CONF"] = "garbage_collection_threshold:0.8"
    # PyTorch CUDA allocator가 사용되지 않는 메모리 블록을 적극적으로 회수하는 임계값을 설정
    # gpu 메모리 사용률이 0.9375를 넘으면 gc 실행

    # torch.cuda.set_per_process_memory_fraction(0.95, device=0)
    # reserved 메모리 크기가 계속 커져서 공유메모리까지 점유하지 않도록 0.95설정(최대 15.4GB)
    # @@@ train loop 내에서 적극적으로 reserved 메모리 크기 확인하고 줄이는 방식만 사용하도록 여기는 주석처리

    # torch.cuda.empty_cache()
    # torch.cuda.synchronize()
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    if not osp.exists(model_dir):
        os.makedirs(model_dir)

    # batch_size = batch_size

    # val_batch_size = val_batch_size

    # -- early stopping flag
    # patience = patience
    # counter = 0

    # 데이터셋
    if osp.basename(data_dir) == "data_10m.csv":
        print("Loading 10m dataset")
        data_dir_dir = osp.dirname(data_dir)
        test_dir = osp.join(data_dir_dir, "data_test.csv")
        loaders = Loaders_10M(data_path=data_dir, test_path=test_dir, max_token_length=max_token_length, target_tokens=target_tokens, batch_size_train=batch_size, num_workers=num_workers, batch_size_val=val_batch_size, batch_size_test=test_batch_size, val_num_workers=val_num_workers,  tokenizer=tokenizer, seed=seed)
    elif osp.basename(data_dir) == "data_9m.csv":
        print("Loading 9m dataset")
        data_dir_dir = osp.dirname(data_dir)
        test_dir = osp.join(data_dir_dir, "data_test.csv")
        loaders = Loaders_9M(data_path=data_dir, test_path=test_dir, max_token_length=max_token_length, target_tokens=target_tokens, batch_size_train=batch_size, num_workers=num_workers, batch_size_val=val_batch_size, batch_size_test=test_batch_size, val_num_workers=val_num_workers,  tokenizer=tokenizer, seed=seed)
    else:
        print("Loading 1m dataset")
        # loaders = Loaders(data_path=data_dir, max_token_length=max_token_length, batch_size_train=batch_size, num_workers=num_workers, batch_size_val=val_batch_size, batch_size_test=test_batch_size, val_num_workers=val_num_workers, start_idx=start_idx, end_idx=end_idx, padding_idx=padding_idx, unk_idx=unk_idx, seed=seed)
        loaders = Loaders(data_path=data_dir, max_token_length=max_token_length, target_tokens= target_tokens, batch_size_train=batch_size, num_workers=num_workers, batch_size_val=val_batch_size, batch_size_test=test_batch_size, val_num_workers=val_num_workers,  tokenizer=tokenizer, seed=seed)

    data_load_end = datetime.now()
    data_load_time = data_load_end - time_start
    data_load_time = str(data_load_time).split(".")[0]
    print(f"==>> data_load_time: {data_load_time}")

    print("".center(50, "-"))

    # Initialize the model
    model = Transformer(
        src_len_vocab=loaders.len_vocab,
        tgt_len_vocab=loaders.len_vocab,
        start_idx=loaders.start_idx,
        end_idx=loaders.end_idx,
        padding_idx=loaders.padding_idx,
        unk_idx=loaders.unk_idx,
        q_dim=q_dim,
        k_dim=q_dim,
        v_dim=q_dim,
        h_dim=(q_dim * 4),
        visualization=viz,
        grad_cp=grad_cp,
        weight_tying=WeightTying[weight_tying],
        # tie_weights=weight_tying,
        # decouple_src_tgt_embed=decouple_src_tgt_embed,
        # decouple_ffc_tgt_embed=decouple_ffc_tgt_embed,
        # decouple_embed_ffc=decouple_embed_ffc,
    )
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

    if debug:
        # 모든 파라미터에 grad NaN 체크하는 hook 등록
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.register_hook(nan_hook)

    # 임베딩 층의 가중치에 grad NaN 체크하는 hook 등록
    # model.src_embed[0].embed.weight.register_hook(nan_hook)

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # hook 사용 대신 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 사용으로 변경
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # 임베딩 층의 가중치에 그래디언트 클리핑 hook 등록
    # model.src_embed[0].embed.weight.register_hook(clip_grad_embed_hook)

    # 모든 층의 가중치에 그래디언트 클리핑 hook 등록
    # @@@ 학습시간이 2.5 시간 정도 추가되는 문제가 있음
    # for name, param in model.named_parameters():
    #     if name == "src_embed.0.embed.weight":
    #         # print("이건 이미 함")
    #         continue
    #     if param.requires_grad:
    #         param.register_hook(clip_grad_hook)
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        # lr=1,
        # betas=(0.9, 0.999),
        # annotated transformer 값 따라하기
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    if adjusted:
        print("using adjusted rate func")
    else:
        print("using not-adjusted rate func")

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda step: rate(step, model_size=q_dim, factor=learning_factor, warmup=warmup_steps, adjusted=adjusted))
    # 새 rate 함수로 변경
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda step: rate(step, warmup=warmup_steps, total_steps=total_steps, decay_rate=decay_rate))

    if mp:
        # mixed precision scaler 초기화
        mp_scaler = torch.amp.GradScaler()
    
    # pth 불러오는 경우
    if resume_name:
        optimizer.load_state_dict(load_dict["optimizer_state_dict"])
        scheduler.load_state_dict(load_dict["scheduler_state_dict"])
        if mp:
            mp_scaler.load_state_dict(load_dict["mp_scaler_state_dict"])


    criterion = LabelSmoothing(size=loaders.len_vocab, padding_idx=loaders.padding_idx, smoothing=label_smoothing)

    print(f"Start training..")

    print("".center(50, "-"))
    print("".center(50, "-"))
    print("".center(50, "-"))

    wandb_log_name = wandb_run_name + "_" + train_start

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
        name=wandb_log_name,
        mode=wandb_mode,
    )

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # @@@ 메모리 누수 원인이 될 수 있음?
    wandb.watch((model,))
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    best_loss = np.inf
    # best_bleu = 0
    if resume_name:
        best_loss = float(load_dict["best_val_token_loss"])

    for epoch in range(max_epoch):
        print("".center(50, "-"))
        print("".center(50, "-"))
        print(f"Start train #{epoch+1:2d}")
        epoch_start = datetime.now()

        if mp:
            epoch_loss, epoch_mean_batch_loss, epoch_mean_token_loss, epoch_total_tokens, epoch_total_tokens_input = train_loop_with_mp(
                loaders,
                model,
                criterion,
                optimizer,
                scheduler,
                mp_scaler,
                accum_steps,
                device,
                wandb_mode,
                train_start,
                train_break,
                debug,
            )
        else:
            epoch_loss, epoch_mean_batch_loss, epoch_mean_token_loss, epoch_total_tokens, epoch_total_tokens_input = train_loop(
                loaders,
                model,
                criterion,
                optimizer,
                scheduler,
                device,
                wandb_mode,
                train_start,
                train_break,
                debug,
            )

        wandb_epoch_dict = {
            "train_batch_loss": epoch_mean_batch_loss,
            "train_token_loss": epoch_mean_token_loss,
        }

        if not train_break:
            wandb.log(wandb_epoch_dict)


        train_end = datetime.now()
        train_time = train_end - epoch_start
        train_time = str(train_time).split(".")[0]
        print("".center(50, "-"))
        print("".center(50, "-"))
        print(
            f"==>> epoch {epoch+1} train_time: {train_time}\nepoch_total_tokens: {epoch_total_tokens}\nepoch_total_tokens_input: {epoch_total_tokens_input}"
        )
        print("".center(50, "-"))
        print(
            f"mean_batch_loss: {round(epoch_mean_batch_loss,4)}\n mean_token_loss: {round(epoch_mean_token_loss,4)}"
        )

        # if (epoch + 1) % save_interval == 0:
        if (epoch + 1) % save_interval == 0 and not train_break:
            
            ckpt_fpath = osp.join(model_dir, f"tf_koren_{wandb_log_name}_latest.pth")

            states = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),  # 모델의 state_dict 저장
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "args": args_dicts,
                "best_val_token_loss": best_loss,
            }
            if mp:
                states["mp_scaler_state_dict"] = mp_scaler.state_dict()

            torch.save(states, ckpt_fpath)
        
        # train loop 종료 후 캐시 정리
        gc.collect()    # 1. Python GC 실행 (CPU 메모리)
        torch.cuda.empty_cache()      # 2. 사용 안하는 GPU 메모리 캐시 해제 요청
        torch.cuda.synchronize()      # 3. GPU가 2번 작업(torch.cuda.empty_cache()) 및 다른 비동기 작업들을 완료할 때까지 대기
        # torch.cuda.synchronize()의 역할: GPU의 모든 비동기 작업(CUDA kernel)이 완료될 때까지 CPU가 대기하게 만든다
        # synchronize가 없으면 torch.cuda.empty_cache()가 비동기로 호출되고 나서
        # 진행중인 GPU 캐시 정리 작업을 기다리지 않고 뒤의 val 루프나 다음 에폭 코드가 호출되어
        # 캐시 정리가 불완전하게 되므로 메모리 해제 실패 및 누적 문제 발생

        # validation 주기에 따라 loss 또는 평가메트릭을 계산하고 best model을 저장
        if (epoch + 1) % val_interval == 0:

            # val 또는 test 과정에 원문, 번역문 정답, 번역문 예측 결과 저장하기
            print("".center(50, "-"))
            print("".center(50, "-"))
            print(f"Start validation #{epoch+1:2d}")
            val_start = datetime.now()
            
            if mp:
                val_loss, val_mean_batch_loss, val_mean_token_loss, val_total_tokens, val_total_tokens_input = val_loop_with_mp(
                    loaders,
                    model,
                    criterion,
                    device,
                    # wandb_mode,
                    val_break,
                )
            else:
                val_loss, val_mean_batch_loss, val_mean_token_loss, val_total_tokens, val_total_tokens_input = val_loop(
                    loaders,
                    model,
                    criterion,
                    device,
                    # wandb_mode,
                    val_break,
                )


            wandb_val_dict = {
                "val_batch_loss": val_mean_batch_loss,
                "val_token_loss": val_mean_token_loss,
            }

            if not val_break:
                wandb.log(wandb_val_dict)

            val_end = datetime.now()
            val_time = val_end - val_start
            val_time = str(val_time).split(".")[0]
            print("".center(50, "-"))
            print("".center(50, "-"))
            print(
                f"==>> epoch {epoch+1} validation time: {val_time}\nval_total_tokens: {val_total_tokens}\nval_total_tokens_input: {val_total_tokens_input}"
            )
            print("".center(50, "-"))
            print(
                f"val_mean_batch_loss: {round(val_mean_batch_loss,4)}\n val_mean_token_loss: {round(val_mean_token_loss,4)}"
            )

            # if best_loss > val_mean_token_loss:
            if best_loss > val_mean_token_loss and not val_break:
                print("".center(50, "-"))
                print(
                    f"Best val token loss performance at epoch: {epoch + 1}, {best_loss:.4f} -> {val_mean_token_loss:.4f}"
                )
                print(f"Save model in {model_dir}")
                states = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),  # 모델의 state_dict 저장
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "args": args_dicts,
                    "best_val_token_loss": val_mean_token_loss,
                }
                if mp:
                    states["mp_scaler_state_dict"] = mp_scaler.state_dict()

                best_ckpt_fpath = osp.join(
                    model_dir, f"tf_koren_{wandb_log_name}_best_val_token_loss.pth"
                )
                torch.save(states, best_ckpt_fpath)
                best_loss = val_mean_token_loss
            #     counter = 0
            # else:
            #     counter += 1

        # val loop 종료 후 캐시 정리
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if (epoch + 1) % test_interval == 0 or (epoch + 1) == max_epoch:
            print("".center(50, "-"))
            print("".center(50, "-"))
            print(f"Start Test #{epoch+1:2d}")
            test_start = datetime.now()

            if mp:
                result, result_per_cat, test_total_tokens, test_total_tokens_input = test_loop_with_mp(
                    loaders,
                    model,
                    device,
                    viz,
                    image_dir,
                    wandb_log_name,
                    test_break,
                )
            else:
                result, result_per_cat, test_total_tokens, test_total_tokens_input = test_loop(
                    loaders,
                    model,
                    device,
                    viz,
                    image_dir,
                    wandb_log_name,
                    test_break,
                )

            test_bleu = result['bleu']
            test_chrf = result['chrf']
            # test_ter = result['ter']
            test_meteor = result['meteor']
            # test_bertscore_f1 = result['bertscore_f1']
            # test_bertscore_precision = result['bertscore_precision']
            # test_bertscore_recall = result['bertscore_recall']


            test_end = datetime.now()
            test_time = test_end - test_start
            test_time = str(test_time).split(".")[0]
            print("".center(50, "-"))
            print("".center(50, "-"))
            print(
                f"==>> test time: {test_time}\ntest_total_tokens: {test_total_tokens}\ntest_total_tokens_input: {test_total_tokens_input}"
            )
            print("".center(50, "-"))
            print(f"test_bleu: {test_bleu}")
            print(f"test_chrf: {test_chrf}")
            # print(f"test_ter: {test_ter}")
            print(f"test_meteor: {test_meteor}")
            # print(f"test_bertscore_f1: {test_bertscore_f1}")
            # print(f"test_bertscore_precision: {test_bertscore_precision}")
            # print(f"test_bertscore_recall: {test_bertscore_recall}")

            wandb_test_dict = {
                "test_bleu": test_bleu,
                "test_chrf": test_chrf,
                # "test_ter": test_ter,
                "test_meteor": test_meteor,
                # "test_bertscore_f1": test_bertscore_f1,
                # "test_bertscore_precision": test_bertscore_precision,
                # "test_bertscore_recall": test_bertscore_recall,
            }

            cat_names = ["구어체", "대화체", "문어체_뉴스", "문어체_한국문화", "문어체_조례", "문어체_지자체웹사이트"]
            for i in range(6):
                print("".center(50, "-"))
                bleu_name = f'test_bleu_{i}_{cat_names[i]}'
                bleu_key = f'bleu_{i}'
                chrf_name = f'test_chrf_{i}_{cat_names[i]}'
                chrf_key = f'chrf_{i}'
                meteor_name = f'test_meteor_{i}_{cat_names[i]}'
                meteor_key = f'meteor_{i}'
                # bert_f1_name = f'test_bertscore_f1_{i}_{cat_names[i]}'
                # bert_f1_key = f'bertscore_f1_{i}'
                # bert_precision_name = f'test_bertscore_precision_{i}_{cat_names[i]}'
                # bert_precision_key = f'bertscore_precision_{i}'
                # bert_recall_name = f'test_bertscore_recall_{i}_{cat_names[i]}'
                # bert_recall_key = f'bertscore_recall_{i}'
                print(bleu_name + f": {result_per_cat[bleu_key]}")
                print(chrf_name + f": {result_per_cat[chrf_key]}")
                print(meteor_name + f": {result_per_cat[meteor_key]}")
                # print(bert_f1_name + f": {result_per_cat[bert_f1_key]}")
                # print(bert_precision_name + f": {result_per_cat[bert_precision_key]}")
                # print(bert_recall_name + f": {result_per_cat[bert_recall_key]}")

                wandb_test_dict[bleu_name] = result_per_cat[bleu_key]
                wandb_test_dict[chrf_name] = result_per_cat[chrf_key]
                wandb_test_dict[meteor_name] = result_per_cat[meteor_key]
                # wandb_test_dict[bert_f1_name] = result_per_cat[bert_f1_key]
                # wandb_test_dict[bert_precision_name] = result_per_cat[bert_precision_key]
                # wandb_test_dict[bert_recall_name] = result_per_cat[bert_recall_key]


            wandb.log(wandb_test_dict)

        # test loop 종료 후 캐시 정리
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        
        epoch_end = datetime.now()
        epoch_time = epoch_end - epoch_start
        epoch_time = str(epoch_time).split(".")[0]
        print("".center(50, "-"))
        print("".center(50, "-"))
        print(
            f"==>> epoch {epoch+1} time: {epoch_time}"
        )

        # if counter > patience:
        #     print("Early Stopping...")
        #     break

    print("".center(50, "-"))
    print("".center(50, "-"))
    print("".center(50, "-"))

    # val 또는 test 과정에 원문, 번역문 정답, 번역문 예측 결과 저장하기



    time_end = datetime.now()
    total_time = time_end - time_start
    total_time = str(total_time).split(".")[0]
    print(f"==>> total time: {total_time}")

    print("".center(50, "-"))
    print("".center(50, "-"))
    print("".center(50, "-"))


# LambdaLR 스케쥴러에 사용하는 함수
def rate(step, model_size, factor, warmup, adjusted):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """

    if not adjusted:
        if step == 0:
            step = 1

        return factor * (
            model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
        )


    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # @@@ 논문은 450만 문장을 각 step당 2만5천씩 한 epoch에 180 step 정도
    # @@@ 대략 556 epoch (100000/180) 진행
    # 현재 학습은 128만 문장을 batch 8로 대략 16만 step
    # 논문 장비 기준으로는 대략 50~51 step ==> step 비 3200
    # step_m = step / 3200
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # @@@@@@ 계산 오류
    # 배치 크기가 2만 5천인게 아니라 배치당 source 토큰 2만5천, target 토큰 2만 5천개
    # 현재 학습 1281934문장의 총 tgt 토큰 개수 40586486 ==> 문장당 평균 31.66 토큰
    # 배치 8 * 31.66 = 253.28
    # 25000 / 253.28 ~= 98.705 ==> 대략 100배
    # step_m = step / 100

    # 배치 32의 경우 ~ 25배
    step_m = step / 25
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    if step_m < 1:
        step_m = 1

    return factor * (
        model_size ** (-0.5) * min(step_m ** (-0.5), step_m * warmup ** (-1.5))
    )


# def rate(step, warmup, total_steps, decay_rate):
#     if step < warmup:
#         # warmup 구간: 선형 증가
#         return float(step) / float(max(1, warmup))
#     else:
#         # warmup 이후: 지수 감쇠
#         decay_steps = total_steps - warmup
#         decay_progress = (step - warmup) / decay_steps
#         return decay_rate ** decay_progress


def main(args):
    # debug를 위해 CUDA 호출을 CPU 코드와 동기화
    if args.__dict__['debug']:
        print("debug mode enabled")
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    else:
        print("debug mode disabled")

    train(args.__dict__, **args.__dict__)
    # 입력 명령어 pth에 저장하기 위해 args.__dict__ 추가로 받기


if __name__ == "__main__":
    args = parse_args()

    main(args)