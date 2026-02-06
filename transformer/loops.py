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


    epoch_loss = 0
    epoch_total_tokens = 0


    num_batches_train = len(loaders.loader_train)

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # 메모리 증가량 확인용 변수들
    # mem_a_start = 0.0
    # mem_r_start = 0.0
    # mem_a_to = 0.0
    # mem_r_to = 0.0
    # mem_a_outloss = 0.0
    # mem_r_outloss = 0.0
    # mem_a_backword = 0.0
    # mem_r_backword = 0.0
    # mem_a_gradcheck = 0.0
    # mem_r_gradcheck = 0.0
    # mem_a_log = 0.0
    # mem_r_log = 0.0
    # mem_a_stepupdate = 0.0
    # mem_r_stepupdate = 0.0
    # mem_a_end = 0.0
    # mem_r_end = 0.0
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    for step, batch in tqdm(
        enumerate(loaders.loader_train),
        total=num_batches_train,
    ):
        # batch는 'kor', 'en', 'cat', 'input_ids', 'attention_mask', 'decoder_inputs', 'decoder_mask', 'labels', 'ntokens'들을 키로 가지는 dict

        if train_break:
            break

        optimizer.zero_grad()

        # current_memory_gb = torch.cuda.memory_allocated() / 1024**3
        current_memory_gb = torch.cuda.memory_reserved() / 1024**3

        if current_memory_gb > 15.5: 
            print(f"Step {step}: Memory {current_memory_gb:.2f}GB > {15.5}GB, cleaning...")

            # print(f"==>> mem_a_start: {mem_a_start}")
            # print(f"==>> mem_r_start: {mem_r_start}")
            # print(f"==>> mem_a_to: {mem_a_to}")
            # print(f"==>> mem_r_to: {mem_r_to}")
            # print(f"==>> mem_a_outloss: {mem_a_outloss}")
            # print(f"==>> mem_r_outloss: {mem_r_outloss}")
            # print(f"==>> mem_a_backword: {mem_a_backword}")
            # print(f"==>> mem_r_backword: {mem_r_backword}")
            # print(f"==>> mem_a_gradcheck: {mem_a_gradcheck}")
            # print(f"==>> mem_r_gradcheck: {mem_r_gradcheck}")
            # print(f"==>> mem_a_log: {mem_a_log}")
            # print(f"==>> mem_r_log: {mem_r_log}")
            # print(f"==>> mem_a_stepupdate: {mem_a_stepupdate}")
            # print(f"==>> mem_r_stepupdate: {mem_r_stepupdate}")
            # print(f"==>> mem_a_end: {mem_a_end}")
            # print(f"==>> mem_r_end: {mem_r_end}")

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"After cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # mem_a_start = torch.cuda.memory_allocated() / 1024**3
        # mem_r_start = torch.cuda.memory_reserved() / 1024**3
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

        x_masks = batch['attention_mask'].to(device, non_blocking=True)
        gt_masks = batch['decoder_mask'].to(device, non_blocking=True)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # mem_a_to = torch.cuda.memory_allocated() / 1024**3
        # mem_r_to = torch.cuda.memory_reserved() / 1024**3 
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
        # mem_a_outloss = torch.cuda.memory_allocated() / 1024**3
        # mem_r_outloss = torch.cuda.memory_reserved() / 1024**3
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        loss_value = loss.item()
        normalized_loss_value = normalized_loss.item()

        normalized_loss.backward()

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # mem_a_backword = torch.cuda.memory_allocated() / 1024**3
        # mem_r_backword = torch.cuda.memory_reserved() / 1024**3
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
        # mem_a_gradcheck = torch.cuda.memory_allocated() / 1024**3
        # mem_r_gradcheck = torch.cuda.memory_reserved() / 1024**3
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
        # mem_a_log = torch.cuda.memory_allocated() / 1024**3
        # mem_r_log = torch.cuda.memory_reserved() / 1024**3
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        optimizer.step()
        scheduler.step()

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # mem_a_stepupdate = torch.cuda.memory_allocated() / 1024**3
        # mem_r_stepupdate = torch.cuda.memory_reserved() / 1024**3
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        if debug:
            # NaN값이 파라메터이 있는지 확인
            if check_nan_in_parameters(model):
                raise ValueError("NaN detected in model parameters!")

        # with torch.no_grad():
        # @@@ loss.item()은 그래프에서 분리된 순수한 숫자(float)이므로 그래디언트 계산과 무관
        epoch_loss += loss_value

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # mem_a_end = torch.cuda.memory_allocated() / 1024**3
        # mem_r_end = torch.cuda.memory_reserved() / 1024**3
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


    # 배치당 loss 값의 평균 계산
    epoch_mean_batch_loss = epoch_loss / num_batches_train
    # 토큰 한개당 loss 값의 평균 계산
    epoch_mean_token_loss = epoch_loss / epoch_total_tokens if epoch_total_tokens != 0 else 0

    return epoch_loss, epoch_mean_batch_loss, epoch_mean_token_loss, epoch_total_tokens


def val_loop(
        loaders:Loaders,
        model,
        criterion,
        device,
        # wandb_mode,
        val_break=False,
    ):

    model.eval()

    with torch.no_grad():
        val_loss = 0
        val_total_tokens = 0

        num_batches_val = len(loaders.loader_val)

        for step, batch_val in tqdm(
            enumerate(loaders.loader_val), total=num_batches_val
        ):  
            if val_break:
                break

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
        
    # 배치당 loss 값의 평균 계산
    val_mean_batch_loss = val_loss / num_batches_val
    # 토큰 한개당 loss 값의 평균 계산
    val_mean_token_loss = val_loss / val_total_tokens if val_total_tokens != 0 else 0

    return val_loss, val_mean_batch_loss, val_mean_token_loss, val_total_tokens



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

    with torch.no_grad():

        test_total_tokens = 0

        num_batches_test = len(loaders.loader_test)

        for step, batch_test in tqdm(
            enumerate(loaders.loader_test), total=num_batches_test
        ):
            if test_break and step == 2:
                break

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
                    visualize(image_dir, log_name=wandb_log_name, step=step, model=model, loaders=loaders, cat_list=batch_test['cat'], inputs=inputs, preds=preds, labels=labels, n_examples=2)

            loaders.add_batch_to_metrics(preds, labels)
            loaders.add_batch_to_metrics_per_cat(preds, labels, batch_test['cat'])
        
    result = loaders.compute_metrics()

    result_per_cat = loaders.compute_metrics_per_cat()

    return result, result_per_cat, test_total_tokens



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


    epoch_loss = 0
    epoch_total_tokens = 0


    num_batches_train = len(loaders.loader_train)

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # 메모리 증가량 확인용 변수들
    # mem_a_start = 0.0
    # mem_r_start = 0.0
    # mem_a_to = 0.0
    # mem_r_to = 0.0
    # mem_a_outloss = 0.0
    # mem_r_outloss = 0.0
    # mem_a_backword = 0.0
    # mem_r_backword = 0.0
    # mem_a_gradcheck = 0.0
    # mem_r_gradcheck = 0.0
    # mem_a_log = 0.0
    # mem_r_log = 0.0
    # mem_a_stepupdate = 0.0
    # mem_r_stepupdate = 0.0
    # mem_a_log2 = 0.0
    # mem_r_log2 = 0.0
    # mem_a_end = 0.0
    # mem_r_end = 0.0
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    for step, batch in tqdm(
        enumerate(loaders.loader_train),
        total=num_batches_train,
    ):
        # batch는 'kor', 'en', 'cat', 'input_ids', 'attention_mask', 'decoder_inputs', 'decoder_mask', 'labels', 'ntokens'들을 키로 가지는 dict

        if train_break:
            break

        optimizer.zero_grad()
        
        # current_memory_gb = torch.cuda.memory_allocated() / 1024**3
        current_memory_gb = torch.cuda.memory_reserved() / 1024**3

        if current_memory_gb > 15.5: 
            print(f"Step {step}: Memory {current_memory_gb:.2f}GB > {15.5}GB, cleaning...")

            # print(f"==>> mem_a_start: {mem_a_start}")
            # print(f"==>> mem_r_start: {mem_r_start}")
            # print(f"==>> mem_a_to: {mem_a_to}")
            # print(f"==>> mem_r_to: {mem_r_to}")
            # print(f"==>> mem_a_outloss: {mem_a_outloss}")
            # print(f"==>> mem_r_outloss: {mem_r_outloss}")
            # print(f"==>> mem_a_backword: {mem_a_backword}")
            # print(f"==>> mem_r_backword: {mem_r_backword}")
            # print(f"==>> mem_a_gradcheck: {mem_a_gradcheck}")
            # print(f"==>> mem_r_gradcheck: {mem_r_gradcheck}")
            # print(f"==>> mem_a_log: {mem_a_log}")
            # print(f"==>> mem_r_log: {mem_r_log}")
            # print(f"==>> mem_a_stepupdate: {mem_a_stepupdate}")
            # print(f"==>> mem_r_stepupdate: {mem_r_stepupdate}")
            # print(f"==>> mem_a_log2: {mem_a_log2}")
            # print(f"==>> mem_r_log2: {mem_r_log2}")
            # print(f"==>> mem_a_end: {mem_a_end}")
            # print(f"==>> mem_r_end: {mem_r_end}")

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"After cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # mem_a_start = torch.cuda.memory_allocated() / 1024**3
        # mem_r_start = torch.cuda.memory_reserved() / 1024**3
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

        x_masks = batch['attention_mask'].to(device, non_blocking=True)
        gt_masks = batch['decoder_mask'].to(device, non_blocking=True)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # mem_a_to = torch.cuda.memory_allocated() / 1024**3
        # mem_r_to = torch.cuda.memory_reserved() / 1024**3 
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
        # mem_a_outloss = torch.cuda.memory_allocated() / 1024**3
        # mem_r_outloss = torch.cuda.memory_reserved() / 1024**3
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        loss_value = loss.item()
        normalized_loss_value = normalized_loss.item()

        mp_scaler.scale(normalized_loss).backward()

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # mem_a_backword = torch.cuda.memory_allocated() / 1024**3
        # mem_r_backword = torch.cuda.memory_reserved() / 1024**3
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
        # mem_a_gradcheck = torch.cuda.memory_allocated() / 1024**3
        # mem_r_gradcheck = torch.cuda.memory_reserved() / 1024**3
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
        # mem_a_log = torch.cuda.memory_allocated() / 1024**3
        # mem_r_log = torch.cuda.memory_reserved() / 1024**3
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
        # mem_a_stepupdate = torch.cuda.memory_allocated() / 1024**3
        # mem_r_stepupdate = torch.cuda.memory_reserved() / 1024**3
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


        if wandb_mode != "disabled":
            wandb.log({"mp_scaler_scale": mp_scaler.get_scale()})


        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # mem_a_log2 = torch.cuda.memory_allocated() / 1024**3
        # mem_r_log2 = torch.cuda.memory_reserved() / 1024**3
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
        # mem_a_end = torch.cuda.memory_allocated() / 1024**3
        # mem_r_end = torch.cuda.memory_reserved() / 1024**3
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


    # 배치당 loss 값의 평균 계산
    epoch_mean_batch_loss = epoch_loss / num_batches_train
    # 토큰 한개당 loss 값의 평균 계산
    epoch_mean_token_loss = epoch_loss / epoch_total_tokens if epoch_total_tokens != 0 else 0



    return epoch_loss, epoch_mean_batch_loss, epoch_mean_token_loss, epoch_total_tokens


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

    with torch.no_grad():
        val_loss = 0
        val_total_tokens = 0

        num_batches_val = len(loaders.loader_val)

        for step, batch_val in tqdm(
            enumerate(loaders.loader_val), total=num_batches_val
        ):  
            if val_break:
                break

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
        
    # 배치당 loss 값의 평균 계산
    val_mean_batch_loss = val_loss / num_batches_val
    # 토큰 한개당 loss 값의 평균 계산
    val_mean_token_loss = val_loss / val_total_tokens if val_total_tokens != 0 else 0

    return val_loss, val_mean_batch_loss, val_mean_token_loss, val_total_tokens



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

    with torch.no_grad():

        test_total_tokens = 0

        num_batches_test = len(loaders.loader_test)

        for step, batch_test in tqdm(
            enumerate(loaders.loader_test), total=num_batches_test
        ):
            if test_break and step == 2:
                break

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
                    visualize(image_dir, log_name=wandb_log_name, step=step, model=model, loaders=loaders, cat_list=batch_test['cat'], inputs=inputs, preds=preds, labels=labels, n_examples=2)

            loaders.add_batch_to_metrics(preds, labels)
            loaders.add_batch_to_metrics_per_cat(preds, labels, batch_test['cat'])
        
    result = loaders.compute_metrics()

    result_per_cat = loaders.compute_metrics_per_cat()

    return result, result_per_cat, test_total_tokens