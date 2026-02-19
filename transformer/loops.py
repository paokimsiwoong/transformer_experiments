import torch

import math
import os.path as osp

import wandb

from tqdm import tqdm

from dataloader import Loaders
from visualize import visualize
from utils import check_nan_in_parameters

import gc
# gc.collect()로 python 객체 정리

import pynvml
# 실제 gpu 하드웨어 gpu 사용량 확인 라이브러리

from functools import partial

# cache 청소를 실행할 reserved 메모리 크기 기준
MEM_THRESHOLD = 15.0
# @@@ 16이 아니라 15로 두는 이유
# @@@ @@@ nvidia-smi used  >  memory_reserved  ≥  memory_allocated
# @@@ @@@ torch.cuda.memory_reserved() 값이 15.0 이어도
# @@@ @@@ CUDA context(5~800MB), cuBLAS workspaces(libarary workspace) 등등의 
# @@@ @@@ 부수적인 메모리 사용량까지 더해진 실제 메모리 사용량(nvidia-smi used)은 16.0에 근접하거나 넘을 수 있다

# reserved 메모리 크기 기준을 넘긴 상태에서 cache 청소 전 몇 step까지 기다려볼지 정하는 값
MEM_COL_PATIENCE = 1

# 메모리 pre-allocation
PREALLOCATE_BATCH_SIZE = 32
# PREALLOCATE_SEQ_SIZE = 160
PREALLOCATE_SEQ_SIZE = 160
# PREALLOCATE_SEQ_SIZE = 230 # @@@ 그냥 같은 값으로 두는게 메모리는 더 먹어도 더 빠름?
# PREALLOCATE_SEQ_SIZE_GT = 230
PREALLOCATE_SEQ_SIZE_GT = 160
# train set input, label 최대길이
# ==>> df['length'].max(): 157
# ==>> df['length_label'].max(): 224
PREALLOCATE_BATCH_SIZE_VAL = 32
# PREALLOCATE_SEQ_SIZE_VAL = 150
PREALLOCATE_SEQ_SIZE_VAL = 200
PREALLOCATE_SEQ_SIZE_GT_VAL = 200
# val set input, label 최대길이
# ==>> df_val['length'].max(): 149
# ==>> df_val['length_label'].max(): 198
PREALLOCATE_BATCH_SIZE_TEST = 64
# PREALLOCATE_SEQ_SIZE_TEST = 100
PREALLOCATE_SEQ_SIZE_TEST = 150
PREALLOCATE_SEQ_SIZE_GT_TEST = 150
# test set input, label 최대길이
# ==>> df_test['length'].max(): 96
# ==>> df_test['length_label'].max(): 141

def train_loop(
        loaders:Loaders,
        model,
        criterion,
        optimizer,
        scheduler,
        device,
        wandb_mode,
        train_start,
        train_break=False,
        debug=False,
    ):

    model.train()

    # step_precheck_after에서 사용가능하도록 partial 사용
    fn_preallocate_memory = partial(preallocate_memory, model=model, vocab_size=loaders.len_vocab, criterion=criterion, optimizer=optimizer, max_batch_size=PREALLOCATE_BATCH_SIZE, max_seq_len=PREALLOCATE_SEQ_SIZE, max_seq_len_gt=PREALLOCATE_SEQ_SIZE_GT, device=device)

    fn_preallocate_memory()

    epoch_loss = 0
    epoch_total_tokens = 0

    epoch_total_tokens_input = 0


    num_batches_train = len(loaders.loader_train)

    # token based sampler 사용 시,
    # 매 에폭 시작 시점에 set_epoch_indices 메소드 실행 필요
    if loaders.target_tokens is not None:
        loaders.sampler.set_epoch_indices()

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # 메모리 증가량 확인용 변수들
    mem_a_start = 0.0
    mem_r_start = 0.0
    mem_a_to = 0.0
    mem_r_to = 0.0
    mem_a_outloss = 0.0
    mem_r_outloss = 0.0
    mem_a_backword = 0.0
    mem_r_backword = 0.0
    mem_a_gradcheck = 0.0
    mem_r_gradcheck = 0.0
    mem_a_log = 0.0
    mem_r_log = 0.0
    mem_a_stepupdate = 0.0
    mem_r_stepupdate = 0.0
    mem_a_end = 0.0
    mem_r_end = 0.0

    # mem_threshold_touch_count = 0
    force_gc = force_gc_gen()
    # force_gc 함수 생성
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    for step, batch in tqdm(
        enumerate(loaders.loader_train),
        total=num_batches_train,
    ):
        # batch는 'kor', 'en', 'cat', 'input_ids', 'attention_mask', 'decoder_inputs', 'decoder_mask', 'labels', 'ntokens'들을 키로 가지는 dict

        if train_break:
            break

        optimizer.zero_grad()

        precheck = step_precheck(
            step, 
            batch['input_ids'].numel(), 
            batch['decoder_inputs'].numel(),
            PREALLOCATE_BATCH_SIZE,
            PREALLOCATE_SEQ_SIZE,
            PREALLOCATE_SEQ_SIZE_GT,
            batch['input_ids'].size(0),
            batch['decoder_inputs'].size(0)
        )

        if not precheck:
            # reserved 메모리가 MEM_THRESHOLD를 넘으면 gc 실행
            precheck = force_gc()

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        mem_a_start = torch.cuda.memory_allocated() / 1024**3
        mem_r_start = torch.cuda.memory_reserved() / 1024**3
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


        inputs = batch['input_ids'].to(device, non_blocking=True)
        # (batch_size, src_seq_len)

        # teacher forcing에 사용할 gts
        gts = batch['decoder_inputs'].to(device, non_blocking=True)
        # (batch_size, tgt_seq_len)

        # loss 계산에 사용할 정답 레이블
        labels = batch['labels'].to(device, non_blocking=True)
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

        batch_ntokens_input = batch['ntokens_input']
        epoch_total_tokens_input += batch_ntokens_input

        x_masks = batch['attention_mask'].to(device, non_blocking=True)
        gt_masks = batch['decoder_mask'].to(device, non_blocking=True)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        mem_a_to = torch.cuda.memory_allocated() / 1024**3
        mem_r_to = torch.cuda.memory_reserved() / 1024**3 
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

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

        # assert not torch.isnan(loss), "Loss value is NaN!"
        # assert not torch.isinf(loss), "Loss value is inf!"

        if torch.isnan(loss) or torch.isinf(loss):
            print("Loss value is NaN or inf!")
            print("".center(50, "-"))
            print("saving current states")
            fpath = osp.join("/home/paokimsiwoong/workspace/github.com/paokimsiwoong/transformer_experiments/transformer/debug_saves", f"debug_{train_start}_latest.pth")

            states = {
                "batch": batch,
                "inputs": inputs,
                "gts": gts,
                "labels": labels,
                "x_masks": x_masks,
                "gt_masks": gt_masks,
                "out": out,
                "loss": loss,
                "model_state_dict": model.state_dict(),  # 모델의 state_dict 저장
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }

            torch.save(states, fpath)
            print("".center(50, "-"))
            print("states saved")
            print("".center(50, "-"))
            raise ValueError("NaN or inf detected in loss!")
        

        normalized_loss = loss / batch_ntokens

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        mem_a_outloss = torch.cuda.memory_allocated() / 1024**3
        mem_r_outloss = torch.cuda.memory_reserved() / 1024**3
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        loss_value = loss.item()
        normalized_loss_value = normalized_loss.item()

        normalized_loss.backward()

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        mem_a_backword = torch.cuda.memory_allocated() / 1024**3
        mem_r_backword = torch.cuda.memory_reserved() / 1024**3
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


        # @@@ NaN이 발생하는 임베딩층의 파라메터와 grad 값 확인
        weight = model.src_embed[0].embed.weight
        grad = weight.grad
        grad_mean = grad.mean().item() if grad is not None else None
        grad_max = grad.max().item() if grad is not None else None
        norm = grad.norm().item() if grad is not None else None
        
        weight_tgt = model.tgt_embed[0].embed.weight
        grad_tgt = weight_tgt.grad
        grad_mean_tgt = grad_tgt.mean().item() if grad_tgt is not None else None
        grad_max_tgt = grad_tgt.max().item() if grad_tgt is not None else None
        norm_tgt = grad_tgt.norm().item() if grad_tgt is not None else None

        # 마지막 ffc 레이어도 확인
        weight_ffc = model.ffc.weight
        grad_ffc = weight_ffc.grad
        grad_mean_ffc = grad_ffc.mean().item() if grad_ffc is not None else None
        grad_max_ffc = grad_ffc.max().item() if grad_ffc is not None else None
        norm_ffc = grad_ffc.norm().item() if grad_ffc is not None else None
        if model.ffc.bias is not None:
            bias_ffc = model.ffc.bias
            grad_ffc_bias = bias_ffc.grad
            grad_mean_ffc_bias = grad_ffc_bias.mean().item() if grad_ffc_bias is not None else None
            grad_max_ffc_bias = grad_ffc_bias.max().item() if grad_ffc_bias is not None else None
            norm_ffc_bias = grad_ffc_bias.norm().item() if grad_ffc_bias is not None else None
            

        # 그래디언트 클리핑
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        # 임베딩 층 max_norm 값 변경
        # torch.nn.utils.clip_grad_norm_(model.src_embed.parameters(), max_norm=max_norm*5)
        # torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=max_norm)
        # torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=max_norm)


        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        mem_a_gradcheck = torch.cuda.memory_allocated() / 1024**3
        mem_r_gradcheck = torch.cuda.memory_reserved() / 1024**3
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        if wandb_mode != "disabled":
            wandb_step_dict = {
                "step_total_loss": loss_value,
                "step_token_loss": normalized_loss_value,
                "learning_rate": scheduler.get_last_lr()[0],
                "embed_weight_mean": weight.mean().item(),
                "embed_weight_max": weight.max().item(),
                "embed_grad_mean": grad_mean,
                # "embed_grad_mean_clipped": grad.mean().item(),
                "embed_grad_max": grad_max,
                # "embed_grad_max_clipped": grad.max().item(),
                "embed_grad_norm": norm,
                # "embed_grad_norm_clipped": grad.norm().item(),
                "tgt_embed_weight_mean": weight_tgt.mean().item(),
                "tgt_embed_weight_max": weight_tgt.max().item(),
                "tgt_embed_grad_mean": grad_mean_tgt,
                "tgt_embed_grad_max": grad_max_tgt,
                "tgt_embed_grad_norm": norm_tgt,
                "ffc_weight_mean": weight_ffc.mean().item(),
                "ffc_weight_max": weight_ffc.max().item(),
                "ffc_weight_grad_mean": grad_mean_ffc,
                "ffc_weight_grad_max": grad_max_ffc,
                "ffc_weight_grad_norm": norm_ffc,
            }
            if model.ffc.bias is not None:
                wandb_step_dict["ffc_bias_mean"] = bias_ffc.mean().item()
                wandb_step_dict["ffc_bias_max"] = bias_ffc.max().item()
                wandb_step_dict["ffc_bias_grad_mean"] = grad_mean_ffc_bias
                wandb_step_dict["ffc_bias_grad_max"] = grad_max_ffc_bias
                wandb_step_dict["ffc_bias_grad_norm"] = norm_ffc_bias

            wandb.log(wandb_step_dict)


        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        mem_a_log = torch.cuda.memory_allocated() / 1024**3
        mem_r_log = torch.cuda.memory_reserved() / 1024**3
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        optimizer.step()
        scheduler.step()

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        mem_a_stepupdate = torch.cuda.memory_allocated() / 1024**3
        mem_r_stepupdate = torch.cuda.memory_reserved() / 1024**3
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        if debug:
            # NaN값이 파라메터이 있는지 확인
            if check_nan_in_parameters(model):
                raise ValueError("NaN detected in model parameters!")

        # with torch.no_grad():
        # @@@ loss.item()은 그래프에서 분리된 순수한 숫자(float)이므로 그래디언트 계산과 무관
        epoch_loss += loss_value

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        mem_a_end = torch.cuda.memory_allocated() / 1024**3
        mem_r_end = torch.cuda.memory_reserved() / 1024**3
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        if wandb_mode != "disabled":
            wandb_mem_dict = {
                "mem_a_start" : mem_a_start,
                "mem_r_start" : mem_r_start,
                "mem_a_to" : mem_a_to,
                "mem_r_to" : mem_r_to,
                "mem_a_outloss" : mem_a_outloss,
                "mem_r_outloss" : mem_r_outloss,
                "mem_a_backword" : mem_a_backword,
                "mem_r_backword" : mem_r_backword,
                "mem_a_gradcheck" : mem_a_gradcheck,
                "mem_r_gradcheck" : mem_r_gradcheck,
                "mem_a_log" : mem_a_log,
                "mem_r_log" : mem_r_log,
                "mem_a_stepupdate" : mem_a_stepupdate,
                "mem_r_stepupdate" : mem_r_stepupdate,
                "mem_a_end" : mem_a_end,
                "mem_r_end" : mem_r_end,
                "batch_ntokens_input": batch_ntokens_input, 
                "batch_ntokens_label": batch_ntokens, 
            }

            wandb.log(wandb_mem_dict)

        step_precheck_after(
            step,
            precheck,
            fn_preallocate_memory
        )

    # 배치당 loss 값의 평균 계산
    epoch_mean_batch_loss = epoch_loss / num_batches_train
    # 토큰 한개당 loss 값의 평균 계산
    epoch_mean_token_loss = epoch_loss / epoch_total_tokens if epoch_total_tokens != 0 else 0

    return epoch_loss, epoch_mean_batch_loss, epoch_mean_token_loss, epoch_total_tokens, epoch_total_tokens_input


def val_loop(
        loaders:Loaders,
        model,
        criterion,
        device,
        # wandb_mode,
        val_break=False,
    ):

    model.eval()

    # step_precheck_after에서 사용가능하도록 partial 사용
    fn_preallocate_memory_no_grad = partial(preallocate_memory_no_grad, model=model, vocab_size=loaders.len_vocab, criterion=criterion, max_batch_size=PREALLOCATE_BATCH_SIZE_VAL, max_seq_len=PREALLOCATE_SEQ_SIZE_VAL, max_seq_len_gt=PREALLOCATE_SEQ_SIZE_GT_VAL, device=device)
    fn_preallocate_memory_no_grad()

    with torch.no_grad():
        val_loss = 0
        val_total_tokens = 0

        val_total_tokens_input = 0

        num_batches_val = len(loaders.loader_val)

        # force_gc = force_gc_gen()
        # force_gc 함수 생성

        for step, batch_val in tqdm(
            enumerate(loaders.loader_val), total=num_batches_val
        ):  
            if val_break:
                break

            # reserved 메모리가 MEM_THRESHOLD를 넘으면 gc 실행
            # force_gc()

            precheck = step_precheck(
                step, 
                batch_val['input_ids'].numel(), 
                batch_val['decoder_inputs'].numel(),
                PREALLOCATE_BATCH_SIZE_VAL,
                PREALLOCATE_SEQ_SIZE_VAL,
                PREALLOCATE_SEQ_SIZE_GT_VAL,
                batch_val['input_ids'].size(0),
                batch_val['decoder_inputs'].size(0)
            )

            inputs = batch_val['input_ids'].to(device, non_blocking=True)
            # (batch_size, src_seq_len)

            # teacher forcing에 사용할 gts
            gts = batch_val['decoder_inputs'].to(device, non_blocking=True)
            # (batch_size, tgt_seq_len)

            # loss 계산에 사용할 정답 레이블
            labels = batch_val['labels'].to(device, non_blocking=True)
            # (batch_size, tgt_seq_len)

            batch_val_ntokens = batch_val['ntokens']

            val_total_tokens += batch_val_ntokens

            batch_val_ntokens_input = batch_val['ntokens_input']

            val_total_tokens_input += batch_val_ntokens_input

            x_masks = batch_val['attention_mask'].to(device, non_blocking=True)
            gt_masks = batch_val['decoder_mask'].to(device, non_blocking=True)

            out = model(inputs, gts, x_masks, gt_masks)
            # out.size는 (b, tgt_seq_len, tgt_len_vocab)

            # loss 계산 시에는 out은 (b * tgt_seq_len, tgt_len_vocab)
            # labels는 (b * tgt_seq_len)로 변경 후 입력
            loss = criterion(out.contiguous().view(-1, out.size(-1)), labels.contiguous().view(-1))

            # normalized_loss = loss / batch_val_ntokens

            loss_value = loss.item()
            # normalized_loss_value = normalized_loss.item()

            # if wandb_mode != "disabled":
            #     wandb_val_step_dict = {
            #         "val_step_total_loss": loss_value,
            #         "val_step_token_loss": normalized_loss_value,
            #     }

            #     wandb.log(wandb_val_step_dict)

            val_loss += loss_value

            step_precheck_after(
                step,
                precheck,
                fn_preallocate_memory_no_grad
            )        

    # 배치당 loss 값의 평균 계산
    val_mean_batch_loss = val_loss / num_batches_val
    # 토큰 한개당 loss 값의 평균 계산
    val_mean_token_loss = val_loss / val_total_tokens if val_total_tokens != 0 else 0

    return val_loss, val_mean_batch_loss, val_mean_token_loss, val_total_tokens, val_total_tokens_input



def test_loop(
        loaders:Loaders,
        model,
        device,
        viz,
        image_dir,
        wandb_log_name,
        test_break=False,
    ):

    # 카테고리별 메트릭 초기화
    loaders.init_metrics_per_cat()

    model.eval()

    # step_precheck_after에서 사용가능하도록 partial 사용
    # fn_preallocate_memory_inference = partial(preallocate_memory_inference, model=model, vocab_size=loaders.len_vocab, max_batch_size=PREALLOCATE_BATCH_SIZE_TEST, max_seq_len=PREALLOCATE_SEQ_SIZE_TEST, max_seq_len_gt=PREALLOCATE_SEQ_SIZE_GT_TEST, device=device)
    # fn_preallocate_memory_inference()

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # 루프 중간에 print를 하면 cpu가 gpu에 sync 요청을 해 속도가 느려진다
    # https://medium.com/@varuntej07/why-pytorch-wastes-your-gpu-memory-on-purpose-and-why-thats-brilliant-0a76899797fb
    # => visualize 내부에서 print를 바로 하지말고 str을 저장했다가 모든 step이 끝나면 출력하도록 변경하기
    viz_texts = []
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    with torch.no_grad():

        test_total_tokens = 0

        test_total_tokens_input = 0

        num_batches_test = len(loaders.loader_test)

        force_gc = force_gc_gen()

        for step, batch_test in tqdm(
            enumerate(loaders.loader_test), total=num_batches_test
        ):
            if test_break and step == 2:
                break

            # reserved 메모리가 MEM_THRESHOLD를 넘으면 gc 실행
            force_gc()

            # precheck = step_precheck(
            #     step, 
            #     batch_test['input_ids'].numel(), 
            #     batch_test['decoder_inputs'].numel(),
            #     PREALLOCATE_BATCH_SIZE_TEST,
            #     PREALLOCATE_SEQ_SIZE_TEST,
            #     PREALLOCATE_SEQ_SIZE_GT_TEST,
            #     batch_test['input_ids'].size(0),
            #     batch_test['decoder_inputs'].size(0)
            # )   

            inputs = batch_test['input_ids'].to(device, non_blocking=True)
            # (batch_size, src_seq_len)

            # teacher forcing에 사용할 gts
            # @@@ test 과정에서는 필요 없음
            # gts = batch_test['decoder_inputs'].to(device)
            # (batch_size, tgt_seq_len)

            # loss 계산에 사용할 정답 레이블
            labels = batch_test['labels'].to(device, non_blocking=True)
            # (batch_size, tgt_seq_len)

            batch_test_ntokens = batch_test['ntokens']

            test_total_tokens += batch_test_ntokens
            
            batch_test_ntokens_input = batch_test['ntokens_input']

            test_total_tokens_input += batch_test_ntokens_input

            x_masks = batch_test['attention_mask'].to(device, non_blocking=True)
            # @@@ test 과정에서는 필요 없음
            # gt_masks = batch_test['decoder_mask'].to(device)

            # preds = model.inference(inputs, x_masks, min(labels.size(-1) * 2, labels.size(-1) + 5))
            preds = model.inference(inputs, x_masks, labels.size(-1) * 2, testing=True)
            # (batch_size, pred_seq_len)

            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # result = loaders.compute_metrics(preds, labels)

            # val_bleu += result['bleu']
            # bleu는 배치별로 계산하지 않고 전체 예측 문장과 전체 정답 문장을 모아서 한꺼번에 BLEU를 계산하는 방식을 사용
            # 이렇게 해야 문맥과 길이 등이 고려된 전체적인 BLEU 점수를 정확하게 측정할 수 있다
            if viz:
                if step % (num_batches_test // 3) == 0:
                    visualize(image_dir, log_name=wandb_log_name, step=step, model=model, loaders=loaders, cat_list=batch_test['cat'], inputs=inputs, preds=preds, labels=labels, n_examples=2, texts=viz_texts)

            loaders.add_batch_to_metrics(preds, labels)
            loaders.add_batch_to_metrics_per_cat(preds, labels, batch_test['cat'])

            # step_precheck_after(
            #     step,
            #     precheck,
            #     fn_preallocate_memory_inference
            # )    

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    for t in viz_texts:
        print(t)
    # visualize 함수에서 생성된 text들 출력
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    result = loaders.compute_metrics()

    result_per_cat = loaders.compute_metrics_per_cat()

    return result, result_per_cat, test_total_tokens, test_total_tokens_input



def train_loop_with_mp(
        loaders:Loaders,
        model,
        criterion,
        optimizer,
        scheduler,
        mp_scaler:torch.amp.GradScaler,
        device,
        wandb_mode,
        train_start,
        train_break=False,
        debug=False,
    ):

    model.train()

    # preallocate_memory(model, loaders.len_vocab, criterion, optimizer, max_batch_size=loaders.batch_size_train, max_seq_len=loaders.max_token_length, device=device)
    # 32*512일 경우 Peak memory: 23.6GB
    # preallocate_memory(model, loaders.len_vocab, criterion, optimizer, max_batch_size=loaders.batch_size_train, max_seq_len=(loaders.max_token_length // 2), device=device)
    # 32*256일 경우 Peak memory: 12.1GB (실 사용량 15.0)

    # preallocate_memory(model, loaders.len_vocab, criterion, optimizer, max_batch_size=PREALLOCATE_BATCH_SIZE, max_seq_len=PREALLOCATE_SEQ_SIZE, device=device)

    # step_precheck_after에서 사용가능하도록 partial 사용
    fn_preallocate_memory = partial(preallocate_memory, model=model, vocab_size=loaders.len_vocab, criterion=criterion, optimizer=optimizer, max_batch_size=PREALLOCATE_BATCH_SIZE, max_seq_len=PREALLOCATE_SEQ_SIZE, max_seq_len_gt=PREALLOCATE_SEQ_SIZE_GT, device=device)

    fn_preallocate_memory()


    epoch_loss = 0
    epoch_total_tokens = 0

    epoch_total_tokens_input = 0


    num_batches_train = len(loaders.loader_train)

    # token based sampler 사용 시,
    # 매 에폭 시작 시점에 set_epoch_indices 메소드 실행 필요
    if loaders.target_tokens is not None:
        loaders.sampler.set_epoch_indices()

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # 메모리 증가량 확인용 변수들
    mem_a_start = 0.0
    mem_r_start = 0.0
    mem_a_to = 0.0
    mem_r_to = 0.0
    mem_a_outloss = 0.0
    mem_r_outloss = 0.0
    mem_a_backword = 0.0
    mem_r_backword = 0.0
    mem_a_gradcheck = 0.0
    mem_r_gradcheck = 0.0
    mem_a_log = 0.0
    mem_r_log = 0.0
    mem_a_stepupdate = 0.0
    mem_r_stepupdate = 0.0
    mem_a_log2 = 0.0
    mem_r_log2 = 0.0
    mem_a_end = 0.0
    mem_r_end = 0.0

    # mem_threshold_touch_count = 0
    force_gc = force_gc_gen()
    # force_gc 함수 생성
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    for step, batch in tqdm(
        enumerate(loaders.loader_train),
        total=num_batches_train,
    ):
        # batch는 'kor', 'en', 'cat', 'input_ids', 'attention_mask', 'decoder_inputs', 'decoder_mask', 'labels', 'ntokens'들을 키로 가지는 dict

        if train_break:
            break

        # if step == 900:
        #     break

        optimizer.zero_grad()
        

        precheck = step_precheck(
            step, 
            batch['input_ids'].numel(), 
            batch['decoder_inputs'].numel(),
            PREALLOCATE_BATCH_SIZE,
            PREALLOCATE_SEQ_SIZE,
            PREALLOCATE_SEQ_SIZE_GT,
            batch['input_ids'].size(0),
            batch['decoder_inputs'].size(0)
        )

        if not precheck:
            # reserved 메모리가 MEM_THRESHOLD를 넘으면 gc 실행
            precheck = force_gc()

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        mem_a_start = torch.cuda.memory_allocated() / 1024**3
        mem_r_start = torch.cuda.memory_reserved() / 1024**3
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        inputs = batch['input_ids'].to(device, non_blocking=True)
        # print(f"==>> inputs.shape: {inputs.shape}")
        # (batch_size, src_seq_len)

        # teacher forcing에 사용할 gts
        gts = batch['decoder_inputs'].to(device, non_blocking=True)
        # (batch_size, tgt_seq_len)

        # loss 계산에 사용할 정답 레이블
        labels = batch['labels'].to(device, non_blocking=True)
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

        batch_ntokens_input = batch['ntokens_input']

        epoch_total_tokens_input += batch_ntokens_input

        x_masks = batch['attention_mask'].to(device, non_blocking=True)
        gt_masks = batch['decoder_mask'].to(device, non_blocking=True)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        mem_a_to = torch.cuda.memory_allocated() / 1024**3
        mem_r_to = torch.cuda.memory_reserved() / 1024**3 
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


        with torch.amp.autocast(device_type=device):
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

            # assert not torch.isnan(loss), "Loss value is NaN!"

            # if torch.isnan(loss) or torch.isinf(loss):
            #     print("Loss value is NaN or inf!")
            #     print("".center(50, "-"))
            #     print("saving current states")
            #     fpath = osp.join("/home/paokimsiwoong/workspace/github.com/paokimsiwoong/transformer_experiments/transformer/debug_saves", f"debug_{train_start}_latest.pth")

            #     states = {
            #         "batch": batch,
            #         "inputs": inputs,
            #         "gts": gts,
            #         "labels": labels,
            #         "x_masks": x_masks,
            #         "gt_masks": gt_masks,
            #         "out": out,
            #         "loss": loss,
            #         "model_state_dict": model.state_dict(),  # 모델의 state_dict 저장
            #         "optimizer_state_dict": optimizer.state_dict(),
            #         "scheduler_state_dict": scheduler.state_dict(),
            #         "mp_scaler_state_dict": mp_scaler.state_dict(),
            #     }

            #     torch.save(states, fpath)
            #     print("".center(50, "-"))
            #     print("states saved")
            #     print("".center(50, "-"))
            #     raise ValueError("NaN or inf detected in loss!")

            normalized_loss = loss / batch_ntokens

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        mem_a_outloss = torch.cuda.memory_allocated() / 1024**3
        mem_r_outloss = torch.cuda.memory_reserved() / 1024**3
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        loss_value = loss.item()
        normalized_loss_value = normalized_loss.item()

        mp_scaler.scale(normalized_loss).backward()

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        mem_a_backword = torch.cuda.memory_allocated() / 1024**3
        mem_r_backword = torch.cuda.memory_reserved() / 1024**3
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@ 실제 grad 값을 로그하거나 그래디언트 클리핑을 진행하기 위해서는 scale 된 grad 값들을 scaling factor로 다시 나누어주어야 한다
        mp_scaler.unscale_(optimizer)
        # @@@ mp_scaler.step(optimizer)는 앞에 mp_scaler.unscale_(optimizer)이 없으면 
        # @@@ 알아서 mp_scaler.unscale_(optimizer)를 내부에서 실행하지만 
        # @@@ 앞에서 명시적으로 실행하면 파라메터 업데이트만 실행
        # @@@ @@@ unscale_ 주석 확인
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


        # @@@ NaN이 발생하는 임베딩층의 파라메터와 grad 값 확인
        weight = model.src_embed[0].embed.weight
        grad = weight.grad
        grad_mean = grad.mean().item() if grad is not None else None
        grad_max = grad.max().item() if grad is not None else None
        norm = grad.norm().item() if grad is not None else None

        weight_tgt = model.tgt_embed[0].embed.weight
        grad_tgt = weight_tgt.grad
        grad_mean_tgt = grad_tgt.mean().item() if grad_tgt is not None else None
        grad_max_tgt = grad_tgt.max().item() if grad_tgt is not None else None
        norm_tgt = grad_tgt.norm().item() if grad_tgt is not None else None

        # 마지막 ffc 레이어도 확인
        weight_ffc = model.ffc.weight
        grad_ffc = weight_ffc.grad
        grad_mean_ffc = grad_ffc.mean().item() if grad_ffc is not None else None
        grad_max_ffc = grad_ffc.max().item() if grad_ffc is not None else None
        norm_ffc = grad_ffc.norm().item() if grad_ffc is not None else None
        if model.ffc.bias is not None:
            bias_ffc = model.ffc.bias
            grad_ffc_bias = bias_ffc.grad
            grad_mean_ffc_bias = grad_ffc_bias.mean().item() if grad_ffc_bias is not None else None
            grad_max_ffc_bias = grad_ffc_bias.max().item() if grad_ffc_bias is not None else None
            norm_ffc_bias = grad_ffc_bias.norm().item() if grad_ffc_bias is not None else None
        #     if math.isnan(grad_mean_ffc_bias) or math.isinf(grad_mean_ffc_bias):
        #         grad_mean_ffc_bias = 0
        #     if math.isnan(grad_max_ffc_bias) or math.isinf(grad_max_ffc_bias):
        #         grad_max_ffc_bias = 0
        #     if math.isnan(norm_ffc_bias) or math.isinf(norm_ffc_bias):
        #         norm_ffc_bias = 0

        # if math.isnan(grad_mean) or math.isinf(grad_mean):
        #     grad_mean = 0
        # if math.isnan(grad_max) or math.isinf(grad_max):
        #     grad_max = 0
        # if math.isnan(norm) or math.isinf(norm):
        #     norm = 0
        # if math.isnan(grad_mean_tgt) or math.isinf(grad_mean_tgt):
        #     grad_mean_tgt = 0
        # if math.isnan(grad_max_tgt) or math.isinf(grad_max_tgt):
        #     grad_max_tgt = 0
        # if math.isnan(norm_tgt) or math.isinf(norm_tgt):
        #     norm_tgt = 0
        # if math.isnan(grad_mean_ffc) or math.isinf(grad_mean_ffc):
        #     grad_mean_ffc = 0
        # if math.isnan(grad_max_ffc) or math.isinf(grad_max_ffc):
        #     grad_max_ffc = 0
        # if math.isnan(norm_ffc) or math.isinf(norm_ffc):
        #     norm_ffc = 0


        # 그래디언트 클리핑
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        # 임베딩 층 max_norm 값 변경
        # torch.nn.utils.clip_grad_norm_(model.src_embed.parameters(), max_norm=max_norm*5)
        # torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=max_norm)
        # torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=max_norm)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        mem_a_gradcheck = torch.cuda.memory_allocated() / 1024**3
        mem_r_gradcheck = torch.cuda.memory_reserved() / 1024**3
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


        if wandb_mode != "disabled":
            wandb_step_dict = {
                "step_total_loss": loss_value,
                "step_token_loss": normalized_loss_value,
                "learning_rate": scheduler.get_last_lr()[0],
                "embed_weight_mean": weight.mean().item(),
                "embed_weight_max": weight.max().item(),
                "embed_grad_mean": grad_mean,
                # "embed_grad_mean_clipped": grad.mean().item(),
                "embed_grad_max": grad_max,
                # "embed_grad_max_clipped": grad.max().item(),
                "embed_grad_norm": norm,
                # "embed_grad_norm_clipped": grad.norm().item(),
                "tgt_embed_weight_mean": weight_tgt.mean().item(),
                "tgt_embed_weight_max": weight_tgt.max().item(),
                "tgt_embed_grad_mean": grad_mean_tgt,
                "tgt_embed_grad_max": grad_max_tgt,
                "tgt_embed_grad_norm": norm_tgt,
                "ffc_weight_mean": weight_ffc.mean().item(),
                "ffc_weight_max": weight_ffc.max().item(),
                "ffc_weight_grad_mean": grad_mean_ffc,
                "ffc_weight_grad_max": grad_max_ffc,
                "ffc_weight_grad_norm": norm_ffc,
            }
            if model.ffc.bias is not None:
                wandb_step_dict["ffc_bias_mean"] = bias_ffc.mean().item()
                wandb_step_dict["ffc_bias_max"] = bias_ffc.max().item()
                wandb_step_dict["ffc_bias_grad_mean"] = grad_mean_ffc_bias
                wandb_step_dict["ffc_bias_grad_max"] = grad_max_ffc_bias
                wandb_step_dict["ffc_bias_grad_norm"] = norm_ffc_bias

            wandb.log(wandb_step_dict)


        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        mem_a_log = torch.cuda.memory_allocated() / 1024**3
        mem_r_log = torch.cuda.memory_reserved() / 1024**3
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


        # optimizer.step()
        # mp_scaler로 step을 하면 grad를 구할 때 mp_scaler.scale에서 곱해졌던 scaling factor를
        # 다시 나눈 다음에 step을 진행해 fp32와 동일한 스케일의 그래디언트로 step이 이뤄지게 한다
        mp_scaler.step(optimizer)
        mp_scaler.update()
        # @@@ mp_scaler.step(optimizer) 이후에 grad 값을 로그하면 
        # @@@ 이때는 이미 weight가 업데이트됐기 때문에, step 이후의 grad는 보통 의미가 없음
        # @@@ @@@ gradient clipping 타이밍은 이미 지나간 상태이고
        # @@@ @@@ 대부분 옵티마이저에서 grad를 건드리기 떄문에 실제 grad 값을 로깅할 수 없음

        scheduler.step()
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # GradScaler.step(optimizer)는 scaling factor 조정 중인 초기에 gradient overflow가 발생하면 
        # optimizer.step()을 건너뛸 수 있는데 그 때 scheduler.step()을 호출 하면
        # scheduler.step()이 호출되지만 optimizer는 업데이트 안 된 상태라서
        # UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. 
        # In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  
        # Failure to do this will result in PyTorch skipping the first value of the learning rate schedule.
        # 라는 경고가 뜨게 된다
        # @@@ 그러나 LambdaLR 같은 step-based scheduler는 첫 번째 값 스킵 문제가 없고, 총 step 수를 미리 지정했으므로 실제 성능 영향 거의 없음 ==> 무시해도 문제 없음
        # 실제로 optimizer 업데이트 시에만 lr_scheduler.step()을 하려면 
        # 
        # @@@ prev_scale 초기화 부분 loss 계산 전에 추가
        # prev_scale = mp_scaler.get_scale()
        # with torch.amp.autocast(device_type=device):
        # ...
        # if mp_scaler.get_scale() >= prev_scale:  # 업데이트 성공이면 참
        #     scheduler.step()
        # 
        # 로 코드를 변경
        # @@@ mp_scaler는 업데이트를 성공하면 scale을 유지하거나 1~2배 사이 값으로 증가
        # @@@ 업데이트를 실패하면 backoff_factor(보통 1/2) 만큼 곱해서 scale을 감소 시킨다
        # @@@ mp_scaler.get_scale() >= prev_scale 조건은 scale이 유지됐거나 증가했다는 의미
        # @@@ 따라서 업데이트 성공
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        mem_a_stepupdate = torch.cuda.memory_allocated() / 1024**3
        mem_r_stepupdate = torch.cuda.memory_reserved() / 1024**3
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


        if wandb_mode != "disabled":
            wandb.log({"mp_scaler_scale": mp_scaler.get_scale()})


        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        mem_a_log2 = torch.cuda.memory_allocated() / 1024**3
        mem_r_log2 = torch.cuda.memory_reserved() / 1024**3
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


        if debug:
            # NaN값이 파라메터이 있는지 확인
            if check_nan_in_parameters(model):
                raise ValueError("NaN detected in model parameters!")

        # with torch.no_grad():
        # @@@ loss.item()은 그래프에서 분리된 순수한 숫자(float)이므로 그래디언트 계산과 무관
        epoch_loss += loss_value

        # if step % 100 == 0: 
        #     gc.collect()
        #     torch.cuda.empty_cache()
        #     torch.cuda.synchronize()


        # del out, loss

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        mem_a_end = torch.cuda.memory_allocated() / 1024**3
        mem_r_end = torch.cuda.memory_reserved() / 1024**3
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        if wandb_mode != "disabled":
            wandb_mem_dict = {
                "mem_a_start" : mem_a_start,
                "mem_r_start" : mem_r_start,
                "mem_a_to" : mem_a_to,
                "mem_r_to" : mem_r_to,
                "mem_a_outloss" : mem_a_outloss,
                "mem_r_outloss" : mem_r_outloss,
                "mem_a_backword" : mem_a_backword,
                "mem_r_backword" : mem_r_backword,
                "mem_a_gradcheck" : mem_a_gradcheck,
                "mem_r_gradcheck" : mem_r_gradcheck,
                "mem_a_log" : mem_a_log,
                "mem_r_log" : mem_r_log,
                "mem_a_stepupdate" : mem_a_stepupdate,
                "mem_r_stepupdate" : mem_r_stepupdate,
                "mem_a_log2" : mem_a_log2,
                "mem_r_log2" : mem_r_log2,
                "mem_a_end" : mem_a_end,
                "mem_r_end" : mem_r_end,
                "batch_ntokens_input": batch_ntokens_input, 
                "batch_ntokens_label": batch_ntokens, 
            }

            wandb.log(wandb_mem_dict)

        step_precheck_after(
            step,
            precheck,
            fn_preallocate_memory
        )


    # 배치당 loss 값의 평균 계산
    epoch_mean_batch_loss = epoch_loss / num_batches_train
    # 토큰 한개당 loss 값의 평균 계산
    epoch_mean_token_loss = epoch_loss / epoch_total_tokens if epoch_total_tokens != 0 else 0



    return epoch_loss, epoch_mean_batch_loss, epoch_mean_token_loss, epoch_total_tokens, epoch_total_tokens_input


def val_loop_with_mp(
        loaders:Loaders,
        model,
        criterion,
        device,
        # wandb_mode,
        val_break=False,
    ):

    model.eval()
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # # torch.amp.autocast보다
    # # 모델을 미리 half()로 변환해두고 추론하는게 더빠름
    # # test_pths.ipynb 확인 결과 1.2배
    # model= model.half()
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    # preallocate_memory_no_grad(model, loaders.len_vocab, criterion, max_batch_size=PREALLOCATE_BATCH_SIZE_VAL, max_seq_len=PREALLOCATE_SEQ_SIZE_VAL, device=device)
    # step_precheck_after에서 사용가능하도록 partial 사용
    fn_preallocate_memory_no_grad = partial(preallocate_memory_no_grad, model=model, vocab_size=loaders.len_vocab, criterion=criterion, max_batch_size=PREALLOCATE_BATCH_SIZE_VAL, max_seq_len=PREALLOCATE_SEQ_SIZE_VAL, max_seq_len_gt=PREALLOCATE_SEQ_SIZE_GT_VAL, device=device)
    fn_preallocate_memory_no_grad()

    with torch.no_grad():

        val_loss = 0
        val_total_tokens = 0

        val_total_tokens_input = 0

        num_batches_val = len(loaders.loader_val)

        # mem_threshold_touch_count = 0
        # force_gc = force_gc_gen()
        # force_gc 함수 생성

        for step, batch_val in tqdm(
            enumerate(loaders.loader_val), total=num_batches_val
        ):  
            if val_break:
                break

            # reserved 메모리가 MEM_THRESHOLD를 넘으면 gc 실행
            # force_gc()

            precheck = step_precheck(
                step, 
                batch_val['input_ids'].numel(), 
                batch_val['decoder_inputs'].numel(),
                PREALLOCATE_BATCH_SIZE_VAL,
                PREALLOCATE_SEQ_SIZE_VAL,
                PREALLOCATE_SEQ_SIZE_GT_VAL,
                batch_val['input_ids'].size(0),
                batch_val['decoder_inputs'].size(0)
            )

            inputs = batch_val['input_ids'].to(device, non_blocking=True)
            # (batch_size, src_seq_len)

            # teacher forcing에 사용할 gts
            gts = batch_val['decoder_inputs'].to(device, non_blocking=True)
            # (batch_size, tgt_seq_len)

            # loss 계산에 사용할 정답 레이블
            labels = batch_val['labels'].to(device, non_blocking=True)
            # (batch_size, tgt_seq_len)

            batch_val_ntokens = batch_val['ntokens']

            val_total_tokens += batch_val_ntokens

            batch_val_ntokens_input = batch_val['ntokens_input']

            val_total_tokens_input += batch_val_ntokens_input

            x_masks = batch_val['attention_mask'].to(device, non_blocking=True)
            gt_masks = batch_val['decoder_mask'].to(device, non_blocking=True)

            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            with torch.amp.autocast(device_type=device):
                out = model(inputs, gts, x_masks, gt_masks)
                # out.size는 (b, tgt_seq_len, tgt_len_vocab)

                # loss 계산 시에는 out은 (b * tgt_seq_len, tgt_len_vocab)
                # labels는 (b * tgt_seq_len)로 변경 후 입력
                loss = criterion(out.contiguous().view(-1, out.size(-1)), labels.contiguous().view(-1))

                # normalized_loss = loss / batch_val_ntokens

            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # # @@@ autocast 대신 model.half() 사용해서 추론
            # # model은 .half()로 fp 16 변환 상태
            # # @@@ 인풋들은 long 텐서이므로 half 변환 필요 없음
            # out = model(inputs, gts, x_masks, gt_masks)
            # # out.size는 (b, tgt_seq_len, tgt_len_vocab)

            # # loss 계산 시에는 out은 (b * tgt_seq_len, tgt_len_vocab)
            # # labels는 (b * tgt_seq_len)로 변경 후 입력
            # loss = criterion(out.contiguous().view(-1, out.size(-1)), labels.contiguous().view(-1))
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

            loss_value = loss.item()
            # normalized_loss_value = normalized_loss.item()

            # if wandb_mode != "disabled":
            #     wandb_val_step_dict = {
            #         "val_step_total_loss": loss_value,
            #         "val_step_token_loss": normalized_loss_value,
            #     }

            #     wandb.log(wandb_val_step_dict)

            val_loss += loss_value

            step_precheck_after(
                step,
                precheck,
                fn_preallocate_memory_no_grad
            )

        
    # 배치당 loss 값의 평균 계산
    val_mean_batch_loss = val_loss / num_batches_val
    # 토큰 한개당 loss 값의 평균 계산
    val_mean_token_loss = val_loss / val_total_tokens if val_total_tokens != 0 else 0

    return val_loss, val_mean_batch_loss, val_mean_token_loss, val_total_tokens, val_total_tokens_input



def test_loop_with_mp(
        loaders:Loaders,
        model,
        device,
        viz,
        image_dir,
        wandb_log_name,
        test_break=False,
    ):

    # 카테고리별 메트릭 초기화
    loaders.init_metrics_per_cat()

    model.eval()

    # preallocate_memory_inference(model, loaders.len_vocab, max_batch_size=PREALLOCATE_BATCH_SIZE_TEST, max_seq_len=PREALLOCATE_SEQ_SIZE_TEST, device=device)
    # step_precheck_after에서 사용가능하도록 partial 사용
    # fn_preallocate_memory_inference = partial(preallocate_memory_inference, model=model, vocab_size=loaders.len_vocab, max_batch_size=PREALLOCATE_BATCH_SIZE_TEST, max_seq_len=PREALLOCATE_SEQ_SIZE_TEST, max_seq_len_gt=PREALLOCATE_SEQ_SIZE_GT_TEST, device=device)
    # fn_preallocate_memory_inference()

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # 루프 중간에 print를 하면 cpu가 gpu에 sync 요청을 해 속도가 느려진다
    # https://medium.com/@varuntej07/why-pytorch-wastes-your-gpu-memory-on-purpose-and-why-thats-brilliant-0a76899797fb
    # => visualize 내부에서 print를 바로 하지말고 str을 저장했다가 모든 step이 끝나면 출력하도록 변경하기
    viz_texts = []
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    with torch.no_grad():

        test_total_tokens = 0

        test_total_tokens_input = 0

        num_batches_test = len(loaders.loader_test)

        # mem_threshold_touch_count = 0
        force_gc = force_gc_gen()

        for step, batch_test in tqdm(
            enumerate(loaders.loader_test), total=num_batches_test
        ):
            if test_break and step == 2:
                break
            
            # reserved 메모리가 MEM_THRESHOLD를 넘으면 gc 실행
            force_gc()

            # precheck = step_precheck(
            #     step, 
            #     batch_test['input_ids'].numel(), 
            #     batch_test['decoder_inputs'].numel(),
            #     PREALLOCATE_BATCH_SIZE_TEST,
            #     PREALLOCATE_SEQ_SIZE_TEST,
            #     PREALLOCATE_SEQ_SIZE_GT_TEST,
            #     batch_test['input_ids'].size(0),
            #     batch_test['decoder_inputs'].size(0)
            # )      

            inputs = batch_test['input_ids'].to(device, non_blocking=True)
            # (batch_size, src_seq_len)

            # teacher forcing에 사용할 gts
            # @@@ test 과정에서는 필요 없음
            # gts = batch_test['decoder_inputs'].to(device)
            # (batch_size, tgt_seq_len)

            # loss 계산에 사용할 정답 레이블
            labels = batch_test['labels'].to(device, non_blocking=True)
            # (batch_size, tgt_seq_len)

            batch_test_ntokens = batch_test['ntokens']

            test_total_tokens += batch_test_ntokens

            batch_test_ntokens_input = batch_test['ntokens_input']

            test_total_tokens_input += batch_test_ntokens_input

            x_masks = batch_test['attention_mask'].to(device, non_blocking=True)
            # @@@ test 과정에서는 필요 없음
            # gt_masks = batch_test['decoder_mask'].to(device)
            
            with torch.amp.autocast(device_type=device):
                # preds = model.inference(inputs, x_masks, min(labels.size(-1) * 2, labels.size(-1) + 5))
                preds = model.inference(inputs, x_masks, labels.size(-1) * 2, testing=True)
                # (batch_size, pred_seq_len)

            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # result = loaders.compute_metrics(preds, labels)

            # val_bleu += result['bleu']
            # bleu는 배치별로 계산하지 않고 전체 예측 문장과 전체 정답 문장을 모아서 한꺼번에 BLEU를 계산하는 방식을 사용
            # 이렇게 해야 문맥과 길이 등이 고려된 전체적인 BLEU 점수를 정확하게 측정할 수 있다
            if viz:
                if step % (num_batches_test // 3) == 0:
                    visualize(image_dir, log_name=wandb_log_name, step=step, model=model, loaders=loaders, cat_list=batch_test['cat'], inputs=inputs, preds=preds, labels=labels, n_examples=2, texts=viz_texts)

            loaders.add_batch_to_metrics(preds, labels)
            loaders.add_batch_to_metrics_per_cat(preds, labels, batch_test['cat'])

            # step_precheck_after(
            #     step,
            #     precheck,
            #     fn_preallocate_memory_inference
            # )            
    
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    for t in viz_texts:
        print(t)
    # visualize 함수에서 생성된 text들 출력
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    result = loaders.compute_metrics()

    result_per_cat = loaders.compute_metrics_per_cat()

    return result, result_per_cat, test_total_tokens, test_total_tokens_input


# 학습 시작 전 최대 길이 더미 배치 1 step을 진행해
# 최대 크기의 버퍼를 pre-allocate하도록 강제해서
# 학습 도중 인풋 길이가 가변하더라도 버퍼를 그대로 사용해서 메모리 조각화 및 reserved 메모리 증가 문제를 해결
def preallocate_memory(model, vocab_size, criterion, optimizer, max_batch_size=32, max_seq_len=512, max_seq_len_gt=512, device="cuda"):
    """
    최대 크기 dummy batch로 메모리 미리 할당
    """
    print("Preallocating memory...")
    
    # 최대 크기 더미 입력 생성
    dummy_inputs = torch.randint(0, vocab_size, 
                                   (max_batch_size, max_seq_len), 
                                   device=device)
    dummy_gts = torch.randint(0, vocab_size, 
                                (max_batch_size, max_seq_len_gt), 
                                device=device)
    dummy_x_masks = torch.ones_like(dummy_inputs, device=device)
    dummy_gt_masks = torch.ones_like(dummy_gts, device=device)

    batch_ntokens = max_batch_size * max_seq_len
    
    # 모델 forward + backward (실제 학습 X)
    # model.train()
    
    with torch.amp.autocast(device_type=device):  # FP16으로 메모리 절약
        out = model(dummy_inputs, dummy_gts, dummy_x_masks, dummy_gt_masks)
        loss = criterion(out.contiguous().view(-1, out.size(-1)), dummy_gts.contiguous().view(-1))

        normalized_loss = loss / batch_ntokens
    
    normalized_loss.backward()  # 그래디언트 계산
    
    # optimizer.step() (가중치 업데이트 X)
    optimizer.zero_grad()  # 그래프 해제
    
    # 피크 메모리 기록
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    print("Preallocation complete")
    print(f"Peak memory: {peak_mem:.1f}GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")

    mem_dict = get_gpu_mem()
    print(f"GPU Used: {mem_dict["used_gb"]:.2f}")
    
    # 피크 리셋 (실제 훈련 시작)
    torch.cuda.reset_peak_memory_stats()
    # @@@ 리셋을 하지 않으면 실제 학습 루프의 피크가 아니라 더미 배치로 설정된 피크값이 메모리 통계에 사용되므로 리셋
    # @@@ @@@ pytorch는 torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated(), torch.cuda.memory_reserved(), torch.cuda.max_memory_reserved() 4가지 값 기록
    return peak_mem


def preallocate_memory_no_grad(model, vocab_size, criterion, max_batch_size=32, max_seq_len=512, max_seq_len_gt=512, device="cuda"):
    """
    최대 크기 dummy batch로 메모리 미리 할당
    """
    print("Preallocating no grad memory...")

    with torch.no_grad():
        
        # 최대 크기 더미 입력 생성
        dummy_inputs = torch.randint(0, vocab_size, 
                                    (max_batch_size, max_seq_len), 
                                    device=device)
        dummy_gts = torch.randint(0, vocab_size, 
                                    (max_batch_size, max_seq_len_gt), 
                                    device=device)
        dummy_x_masks = torch.ones_like(dummy_inputs, device=device)
        dummy_gt_masks = torch.ones_like(dummy_gts, device=device)
        
        # 모델 forward + backward (실제 학습 X)
        # model.train()
        
        with torch.amp.autocast(device_type=device):  # FP16으로 메모리 절약
            out = model(dummy_inputs, dummy_gts, dummy_x_masks, dummy_gt_masks)
            loss = criterion(out.contiguous().view(-1, out.size(-1)), dummy_gts.contiguous().view(-1))

    
    # 피크 메모리 기록
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    print("No grad preallocation complete.")
    print(f"Peak memory: {peak_mem:.1f}GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")

    mem_dict = get_gpu_mem()
    print(f"GPU Used: {mem_dict["used_gb"]:.2f}")
    
    # 피크 리셋 (실제 훈련 시작)
    torch.cuda.reset_peak_memory_stats()
    # @@@ 리셋을 하지 않으면 실제 학습 루프의 피크가 아니라 더미 배치로 설정된 피크값이 메모리 통계에 사용되므로 리셋
    # @@@ @@@ pytorch는 torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated(), torch.cuda.memory_reserved(), torch.cuda.max_memory_reserved() 4가지 값 기록
    return peak_mem


def preallocate_memory_inference(model, vocab_size, max_batch_size=32, max_seq_len=512, max_seq_len_gt=512, device="cuda"):
    """
    최대 크기 dummy batch로 메모리 미리 할당
    """
    print("Preallocating inference memory...")

    with torch.no_grad():
    
        # 최대 크기 더미 입력 생성
        dummy_inputs = torch.randint(0, vocab_size, 
                                    (max_batch_size, max_seq_len), 
                                    device=device)
        dummy_labels = torch.randint(0, vocab_size, 
                                    (max_batch_size, max_seq_len_gt), 
                                    device=device)
        dummy_x_masks = torch.ones_like(dummy_inputs, device=device)

        
        # 모델 forward + backward (실제 학습 X)
        # model.train()
        
        with torch.amp.autocast(device_type=device):  # FP16으로 메모리 절약
            preds = model.inference(dummy_inputs, dummy_x_masks, dummy_labels.size(-1) * 2, testing=True)

    
    # 피크 메모리 기록
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    print("Inference preallocation complete.")
    print(f"Peak memory: {peak_mem:.1f}GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")

    mem_dict = get_gpu_mem()
    print(f"GPU Used: {mem_dict["used_gb"]:.2f}")
    
    # 피크 리셋 (실제 훈련 시작)
    torch.cuda.reset_peak_memory_stats()
    # @@@ 리셋을 하지 않으면 실제 학습 루프의 피크가 아니라 더미 배치로 설정된 피크값이 메모리 통계에 사용되므로 리셋
    # @@@ @@@ pytorch는 torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated(), torch.cuda.memory_reserved(), torch.cuda.max_memory_reserved() 4가지 값 기록
    return peak_mem

# pynvml로 실제 메모리 사용량 확인하는 함수
def get_gpu_mem(device_index: int = 0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)  
    # info에서 .total, .used, .free로 bytes 단위 메모리 값 확인 가능
    total = info.total / 1024**3   # GB 단위로 변환
    used  = info.used  / 1024**3
    free  = info.free  / 1024**3
    pynvml.nvmlShutdown()
    return {"total_gb": total, "used_gb": used, "free_gb": free}

# reserved 메모리가 MEM_THRESHOLD를 넘으면 gc 실행하는 함수 생성기
def force_gc_gen():
    mem_threshold_touch_count = 0
    def force_gc():
        nonlocal mem_threshold_touch_count
        current_memory_gb = torch.cuda.memory_reserved() / 1024**3
        if current_memory_gb > MEM_THRESHOLD:
            mem_threshold_touch_count += 1
            if mem_threshold_touch_count >= MEM_COL_PATIENCE:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                mem_threshold_touch_count = 0
                return True
            return False
        return False

    return force_gc

# 현재 step에 할당해야할 텐서가 preallocation 크기보다 커지면 메모리 정리하는 함수
def step_precheck(step, input_numel, gt_numel, pre_batch_size, pre_seq_size, pre_seq_size_gt, step_batch_size, step_label_batch_size):
    if input_numel + gt_numel > (pre_batch_size * pre_seq_size + pre_batch_size * pre_seq_size_gt) :
        print(f"Step {step}: input {input_numel} + gt numels {gt_numel} = {input_numel + gt_numel} > pre-allocate numel {pre_batch_size * pre_seq_size + pre_batch_size * pre_seq_size_gt}, cleaning...")
        print(f"input_batch_size {step_batch_size} label_batch_size {step_label_batch_size}")
        print(f"Memory before cleanup")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        print(f"After cleanup")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

        return True
    
    return False

# step_precheck가 실행된 step 종료 후 다시 설정된 값으로 memory preallocation 실행하는 함수
def step_precheck_after(step, precheck, fn_preallocate):
    if precheck:
        print(f"Step {step} finished:")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        fn_preallocate()

        print(f"After preallocation")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

        
