import torch

import wandb

from tqdm import tqdm

from dataloader import Loaders
from visualize import visualize


def train_loop(
        loaders:Loaders,
        model,
        criterion,
        optimizer,
        scheduler,
        device,
        train_break=False,
    ):

    model.train()


    epoch_loss = 0
    epoch_total_tokens = 0


    num_batches_train = len(loaders.loader_train)

    for step, batch in tqdm(
        enumerate(loaders.loader_train),
        total=num_batches_train,
    ):
        # batch는 'kor', 'en', 'cat', 'input_ids', 'attention_mask', 'decoder_inputs', 'decoder_mask', 'labels', 'ntokens'들을 키로 가지는 dict

        if train_break and step == 2:
            break
        

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

        assert not torch.isnan(loss), "Loss value is NaN!"

        normalized_loss = loss / batch_ntokens

        normalized_loss.backward()

        loss_value = loss.item()
        normalized_loss_value = normalized_loss.item()

        # @@@ NaN이 발생하는 임베딩층의 파라메터와 grad 값 확인
        weight = model.src_embed[0].embed.weight
        grad = weight.grad
        grad_mean = grad.mean().item() if grad is not None else None
        grad_max = grad.max().item() if grad is not None else None
        norm = grad.norm().item() if grad is not None else None

        # 그래디언트 클리핑
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        # 임베딩 층 max_norm 값 변경
        # torch.nn.utils.clip_grad_norm_(model.src_embed.parameters(), max_norm=max_norm*5)
        # torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=max_norm)
        # torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=max_norm)

        # with torch.no_grad():
        # @@@ 텐서.item()은 그래프에서 분리된 순수한 숫자(float)이므로 그래디언트 계산과 무관
        wandb_step_dict = {
            "step_total_loss": loss_value,
            "step_token_loss": normalized_loss_value,
            "learning_rate": scheduler.get_last_lr()[0],
            "embed_weight_mean": weight.mean().item(),
            "embed_weight_max": weight.max().item(),
            "embed_grad_mean": grad_mean,
            "embed_grad_mean_clipped": grad.mean().item(),
            "embed_grad_max": grad_max,
            "embed_grad_max_clipped": grad.max().item(),
            "embed_grad_norm": norm,
            "embed_grad_norm_clipped": grad.norm().item(),
        }

        wandb.log(wandb_step_dict)


        optimizer.step()
        scheduler.step()

        # NaN값이 파라메터이 있는지 확인
        # if check_nan_in_parameters(model):
        #     raise ValueError("NaN detected in model parameters!")

        # with torch.no_grad():
        # @@@ loss.item()은 그래프에서 분리된 순수한 숫자(float)이므로 그래디언트 계산과 무관
        epoch_loss += loss_value


    # 배치당 loss 값의 평균 계산
    epoch_mean_batch_loss = epoch_loss / num_batches_train
    # 토큰 한개당 loss 값의 평균 계산
    epoch_mean_token_loss = epoch_loss / epoch_total_tokens

    return epoch_loss, epoch_mean_batch_loss, epoch_mean_token_loss, epoch_total_tokens


def val_loop(
        loaders:Loaders,
        model,
        device,
        val_break=False,
    ):

    model.eval()

    with torch.no_grad():
        val_total_tokens = 0

        num_batches_val = len(loaders.loader_val)

        for step, batch_val in tqdm(
            enumerate(loaders.loader_val), total=num_batches_val
        ):  
            if val_break and step == 2:
                break

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

            # preds = model.inference(inputs, x_masks, min(labels.size(-1) * 2, labels.size(-1) + 5))
            preds = model.inference(inputs, x_masks, labels.size(-1) * 2)
            # (batch_size, pred_seq_len)

            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # result = loaders.compute_metrics(preds, labels)

            # val_bleu += result['bleu']
            # bleu는 배치별로 계산하지 않고 전체 예측 문장과 전체 정답 문장을 모아서 한꺼번에 BLEU를 계산하는 방식을 사용
            # 이렇게 해야 문맥과 길이 등이 고려된 전체적인 BLEU 점수를 정확하게 측정할 수 있다

            loaders.add_batch_to_metrics(preds, labels)
        
    result = loaders.compute_metrics()

    return result, val_total_tokens



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

            # preds = model.inference(inputs, x_masks, min(labels.size(-1) * 2, labels.size(-1) + 5))
            preds = model.inference(inputs, x_masks, labels.size(-1) * 2)
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