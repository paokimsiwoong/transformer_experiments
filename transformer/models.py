import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import math

import blocks, embeddings, pe

# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
# https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch
# https://cpm0722.github.io/pytorch-implementation/transformer
# https://nlp.seas.harvard.edu/2018/04/03/attention.html
# https://nlp.seas.harvard.edu/annotated-transformer/
# https://metamath1.github.io/blog/posts/gentle-t5-trans/gentle_t5_trans.html


class Encoder(nn.Module):
    def __init__(self, q_dim, h_dim, head_num, block_num, drop_rate, visualization=False):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                blocks.EncodingBlock(q_dim=q_dim, h_dim=h_dim, head_num=head_num, drop_rate=drop_rate, visualization=visualization)
                for i in range(block_num)
            ]
        )

    def forward(self, Q, mask, testing=False):
        out = Q

        for m in self.blocks:
            out = m(out, mask, testing)

        return out


class Decoder(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, h_dim, head_num, block_num, drop_rate, visualization=False):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                blocks.DecodingBlock(
                    q_dim=q_dim, k_dim=k_dim, v_dim=v_dim, h_dim=h_dim, head_num=head_num, drop_rate=drop_rate, visualization=visualization
                )
                for i in range(block_num)
            ]
        )

    def forward(
        self,
        Q,
        K,
        V,
        tgt_mask,
        src_mask,
        testing=False,
    ):
        out = Q

        for m in self.blocks:
            out = m(out, K, V, tgt_mask, src_mask, testing)

        return out

class Transformer(nn.Module):
    def __init__(
        self,
        src_len_vocab,
        tgt_len_vocab,
        # AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en") 기준
        # start_idx는 원래 없지만 .add_special_tokens으로 추가해 65001
        # start_idx=65001,
        # end_idx=0,
        # padding_idx=65000,
        # unk_idx=1,
        # AutoTokenizer.from_pretrained("KETI-AIR/ke-t5-base") 기준
        # start_idx는 원래 없지만 .add_special_tokens으로 추가해 64100
        start_idx=64100,
        end_idx=1,
        padding_idx=0,
        unk_idx=2,
        # start_idx=0,
        # end_idx=1,
        # padding_idx=2,
        # unk_idx=3,
        # sep_idx=4,
        # mask_idx=5,
        max_len = 512 * 2,
        q_dim=512,
        k_dim=512,
        v_dim=512,
        h_dim=2048,
        head_num=8,
        en_block_num=6,
        de_block_num=6,
        drop_rate=0.1,
        tie_weights=True,
        decouple_src_tgt_embed=False,
        visualization=False,
    ):
        super().__init__()

        self.start_idx = start_idx
        self.end_idx = end_idx
        self.padding_idx = padding_idx
        self.unk_idx = unk_idx
        # self.sep_idx = sep_idx
        # self.mask_idx = mask_idx
        self.max_len = max_len

        self.tgt_len_vocab = tgt_len_vocab

        self.tie_weights = tie_weights

        self.src_embed = nn.Sequential(embeddings.Embeddings(src_len_vocab, q_dim, padding_idx), pe.PositionalEncoding(d_model=q_dim, dropout_prob=drop_rate, max_len=max_len))
        self.tgt_embed = nn.Sequential(embeddings.Embeddings(tgt_len_vocab, q_dim, padding_idx), pe.PositionalEncoding(d_model=q_dim, dropout_prob=drop_rate, max_len=max_len))
        # num_embeddings = 단어 집합 개수 (len(vocab))
        # embedding_dim = embed 된 결과 벡터의 차원
        # padding_idx = 패딩 토큰 <PAD> 의 index를 nn.Embedding에 알려주는 인자


        self.encoder = Encoder(
            q_dim=q_dim, h_dim=h_dim, head_num=head_num, block_num=en_block_num, drop_rate=drop_rate, visualization=visualization
        )
        self.decoder = Decoder(
            q_dim=q_dim,
            k_dim=k_dim,
            v_dim=v_dim,
            h_dim=h_dim,
            head_num=head_num,
            block_num=de_block_num,
            drop_rate=drop_rate,
            visualization=visualization,
        )

        
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # 소스/타겟 언어가 다르고 문자계 다르면 (ex. 영어-한국어)
        # embedding/generator 분리 (가중치 tying 비권장)
        # 
        # 소스/타겟 언어가 같거나 매우 유사한 경우
        # Three way weight tying (TWWT: encoder와 decoder, generator까지 모두 공유)가 유리 (메모리 절감, 규제 효과)
        # @@@ 사용하는 토크나이저가 소스/타겟 언어를 단일 vocab으로 토큰화 하는 경우도 weight tying 하는게 유리
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if not tie_weights:
            print("no weight tying")
            self.ffc = nn.Linear(in_features=q_dim, out_features=tgt_len_vocab)
            print("".center(50, "-"))
        else:
            print("weight tying")
            # self.src_embed[0].weight = self.tgt_embed[0].weight
            # @@@ embeddings.Embeddings 안의 self.embed가 nn.Embedding 이다
            if not decouple_src_tgt_embed:
                print("couple src tgt embed")
                self.src_embed[0].embed.weight = self.tgt_embed[0].embed.weight
            else:
                print("decouple src tgt embed")
            self.ffc = nn.Linear(in_features=q_dim, out_features=tgt_len_vocab, bias=False) # embedding에는 bias 없으므로 weight 공유할 ffc도 bias false
            # self.ffc.weight = self.tgt_embed[0].weight
            self.ffc.weight = self.tgt_embed[0].embed.weight
            print("".center(50, "-"))
        # @@@ https://stackoverflow.com/questions/65456174/how-to-use-an-embedding-layer-as-a-linear-layer-in-pytorch
        # @@@ 논문처럼 pre-softmax linear의 weight는 nn.Embedding weight를 공유해서 사용
        # @@@ 여기서 = 는 reference => embed의 weight가 수정되면(역전파 등) ffc의 weight도 수정된다
        # @@@ code_prac 확인
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # .weight는 class 'torch.nn.parameter.Parameter'이고 weight를 바로 수정하려 하면
        # RuntimeError: a view of a leaf Variable that requires grad is being used in an in-place operation. 발생
        # ==> weight를 수동으로 수정하려 할때는 Parameter 클래스인 weight가 아니라
        # class 'torch.Tensor'인 .weight.data를 수정한다
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@ softmax 결과인 x에 0.0이 존재할 수 있어서 loss 계산 내부에서 x.log() 사용 시 -inf 값이 나오는 문제가 있으므로 코드 변경
        # @@@ @@@ ==> 모델 forward 출력에 softmax를 제거하고 loss 계산 내부에서 F.log_softmax() 사용
        # self.softmax = nn.Softmax(dim=-1)
        # decoder 결과를 받아 vocab 단어별 확률을 출력하는 마지막 fc 층
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        # TODO: 파라메터 초기화 부분 추가 필요



    def forward(self, x, gt, x_mask, gt_mask, testing=False):
        # x, gt는 embed 되기전 seq token id들을 담은 long tensor
        # (b, src_seq_len)
        # (b, tgt_seq_len)
        # @@@ gt는 맨끝 end token을 잘라낸 상태

        src_mask = self.make_pad_mask(x_mask)
        tgt_mask = self.make_tgt_mask(gt, gt_mask)

        assert src_mask.size(-1) == x.size(-1)
        assert tgt_mask.size(-1) == gt.size(-1)
        assert tgt_mask.size(-2) == gt.size(-1)

        Q_en = self.src_embed(x)
        # (b, src_seq_len, q_dim)
        # @@ batch안의 각 문장의 원래 길이는 다 다르지만 collate함수를 이용해 dataloader가 batch단위로 내보낼 때
        # @@ batch안의 가장 긴 문장의 길이를 기준으로 그보다 짧은 문장들은 다 padding을 추가해준다

        out_en = self.encoder(Q_en, src_mask, testing)
        # (b, src_seq_len, q_dim)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # embedding oob 확인
        # assert torch.max(gt) <= self.tgt_len_vocab - 1
        # idx_max = torch.max(gt)

        # if idx_max > (self.tgt_len_vocab - 1):
        #     print("idx_max:", idx_max)
        #     raise(ValueError)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        Q_de = self.tgt_embed(gt)
        # (b, tgt_seq_len, q_dim)
        # transformer 학습에는 teacher forcing 기법을 사용 => gt 입력

        out_de = self.decoder(Q_de, out_en, out_en, tgt_mask, src_mask, testing)
        # decoding block 2번째 encoder-decoder attention의 Q,K,V는
        # Q: 1번 Masked self-attention output
        # K,V: encoder output
        # 로 되어 있으므로 out_en을 K,V 자리에 입력
        # (b, tgt_seq_len, q_dim)


        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@ softmax 결과인 x에 0.0이 존재할 수 있어서 loss 계산 내부에서 x.log() 사용 시 -inf 값이 나오는 문제가 있으므로 코드 변경
        # @@@ @@@ ==> 모델 forward 출력에 softmax를 제거하고 loss 계산 내부에서 F.log_softmax() 사용
        return self.ffc(out_de)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # out = self.softmax(self.ffc(out_de))
        # (b, tgt_seq_len, tgt_len_vocab)

        # return out
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    def inference(self, x, x_mask, max_pred_len, testing=False):
        # Greedy decoding
        # TODO: Beam search 구현
        
        # x는 embed 되기전 seq token id들을 담은 long tensor
        # (b, seq_len)

        device = x.device

        n_batch = x.size(0)

        src_mask = self.make_pad_mask(x_mask)

        assert src_mask.size(-1) == x.size(-1)

        Q_en = self.src_embed(x)
        # (b, seq_len, q_dim)
        # @@ batch안의 각 문장의 원래 길이는 다 다르지만 collate함수를 이용해 dataloader가 batch단위로 내보낼 때
        # @@ batch안의 가장 긴 문장의 길이를 기준으로 그보다 짧은 문장들은 다 padding을 추가해준다

        out_en = self.encoder(Q_en, src_mask, testing)
        # (b, seq_len, q_dim)

        ys = torch.zeros(n_batch, 1, device=device).fill_(self.start_idx).type_as(x)
        # (b, 1) 에 start 토큰을 채우기
        # @@@ 텐서 생성 시에 x와 동일한 device에 있도록 반드시 명시

        # for i in range(self.max_len):
        # 5000을 max로 하면 메모리 초과 문제 발생?
        for i in range(max_pred_len):

            tgt_mask = self.make_tgt_mask_inference(ys, self.padding_idx)

            assert tgt_mask.size(-1) == ys.size(-1)

            Q_de = self.tgt_embed(ys)
            # (b, seq_len, q_dim)

            out_de = self.decoder(Q_de, out_en, out_en, tgt_mask, src_mask, testing)
            # decoding block 2번째 encoder-decoder attention의 Q,K,V는
            # Q: 1번 Masked self-attention output
            # K,V: encoder output
            # 로 되어 있으므로 out_en을 K,V 자리에 입력

            # out = self.softmax(self.ffc(out_de))
            # out.size() = (b, seq_len, tgt_len_vocab)
            # 마지막 단어의 결과(dim=1(seq_len)의 마지막)가 새 단어 예측
            # next_token_prob = out[:, -1, :]
            # (b, tgt_len_vocab)
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # @@@ ffc linear에 들어가기 전에 미리 마지막 단어만 남기는게 계산 및 메모리 절약에 도움이 된다
            # # @@@ ffc linear에 매번 out_de 전체를 넣으면 메모리 초과 문제 발생
            # next_token_prob = self.softmax(self.ffc(out_de[:, -1, :]))
            next_token_prob = F.softmax(self.ffc(out_de[:, -1, :]), dim=-1)
            # 마지막 단어의 결과(dim=1(seq_len)의 마지막)가 새 단어 예측
            # out.size() = (b, tgt_len_vocab)
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

            # argmax로 확률이 제일 큰 index 찾기
            next_token = next_token_prob.argmax(dim=-1).unsqueeze(1)
            # (b, 1)

            ys = torch.cat([ys, next_token], dim=1)

            # 모든 문장에 eos 토큰이 생성된 것을 확인하면 for 루프 break
            eos_in_sentence = (ys == self.end_idx).any(dim=1)
            if eos_in_sentence.all():
                break

        # end token 뒷부분 마스킹 처리

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # eos_indices = (ys == self.end_idx).nonzero()
        # mask = torch.zeros_like(ys, device=device)
        # for b, idx in eos_indices:
        #     mask[b, idx+1:] = 1

        # ys.masked_fill_(mask.bool(), value=self.padding_idx)
        # for 루프대신 파이토치 벡터연산을 활용하는 방식으로 변경하기
        # for 루프 방식과 변경된 방식 사이에 메모리나 속도 차이 비교?
        # # batch size 8~64 사이에서는 큰 차이 없음?
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        # (b, seq_len) 크기 ys에서 각 문장별 첫 eos 토큰 위치 찾기
        eos_positions = torch.argmax((ys == self.end_idx).int(), dim=1, keepdim=True)  
        # (b,1)
        # # If there are multiple maximal values then the indices of the first maximal value are returned.
        # # ==> self.end_idx가 문장 안에 여러번 나와도 최초 출현 위치의 idx만 반환한다
        
        # 각 문장별 위치 인덱스 생성 (seq_len)
        seq_indices = torch.arange(ys.size(1), device=device).unsqueeze(0)
        # (1, seq_len)
        # [[0, 1, 2, 3, ...., seq_len - 1]] 형태

        # mask: eos_pos 이후 토큰 True, 아니라면 False
        mask = seq_indices > eos_positions
        # (1, seq_len)인 seq_indices와 (b, 1)인 eos_positions이 broadcast되어 둘다 (b, seq_len) 형태가 된 후 연산
        # ex: 
        # ys의 batch 0번이 5번 인덱스에서 최초로 end_idx가 나올 때
        # seq_indices[0] = [0, 1, 2, 3, 4, 5, ..., seq_len - 1]
        # eos_positions[0] = [5, 5, 5, 5, 5, 5, 5, ....]
        # (seq_indices > eos_positions)[0] = [False, False, False, False, False, False, True, True, True, True, ....]

        ys.masked_fill_(mask, value=self.padding_idx)

        # labels와 동일하게 시작토큰을 제외
        return ys[:, 1:]

    def make_pad_mask(self, x_mask):
        # x_mask: (n_batch, key_seq_len)

        mask = x_mask.unsqueeze(1).unsqueeze(2)  # (n_batch, 1, 1, key_seq_len)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # MHA에서 Q는 (n_batch, n_head, query_seq_len, d_k), K는 (n_batch, n_head, key_seq_len, d_k)인 상태에서
        # Q@K^T / sqrt(d_k) 가 계산되어 (n_batch, n_head, query_seq_len, key_seq_len)인 attention score가 되고
        # 여기서 (n_batch, 1, 1, key_seq_len)인 mask가 broadcasting 되어 
        # (n_batch, n_head, query_seq_len, key_seq_len)인 attention score에 적용
        # ==> query 일반 단어 토큰과 key의 pad 토큰 사이 attention score 계산 값 masking
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        return mask
        
    def make_subsequent_mask(self, query):
        # query: (n_batch, query_seq_len)
        query_seq_len = query.size(1)

        # teacher forcing
        # 훈련 과정에서 query로 gt 전체 문장이 들어가지만
        # i번째 단어를 예측할 때 0~i-1번째 단어와 자기 자신만 보고 미래의 i+1~eos 단어는 보지 않도록 마스킹
        # @@@ ex: 학습 과정에서 query는 SOS I am a student EOS로 전체 정답 문장이 입력된다
        # @@@ 첫번째 단어 SOS의 attention score 계산 시, SOS는 SOS 자기 자신과만 attention score 계산
        # @@@ 두번째로 I의 attention score 계산 시, I 는 SOS와 I와만 attention score 계산
        tril = torch.tril(torch.ones(query_seq_len, query_seq_len, device=query.device))
        # @@@ 텐서 생성 시에 query와 동일한 device에 있도록 반드시 명시
        mask = tril.bool()
        return mask
    
    def make_tgt_mask(self, tgt, tgt_mask):
        pad_mask = self.make_pad_mask(tgt_mask)
        # pad_mask는 (n_batch, 1, 1, tgt_seq_len)꼴
        seq_mask = self.make_subsequent_mask(tgt) 
        # seq_mask는 (tgt_seq_len, tgt_seq_len)꼴

        # pad 부분 mask와 subsequent mask &(and) 연산
        mask = pad_mask & seq_mask
        # @@@ pad_mask와 seq_mask가 (n_batch, 1, tgt_seq_len, tgt_seq_len) 형태로 broadcasting된 뒤 and 연산이 된다
        return mask

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# dataloader의 collate 함수에서 attention 마스크 생성하는 것으로 변경
# inference에서는 ys 마스크 생성에 필요
    def make_pad_mask_inference(self, key, pad_idx=2):
        # key: (n_batch, key_seq_len)
        # key는 embedding 되기 전의 상태로 각 단어가 vocab안의 해당 인덱스로 저장되어 있음

        # pad_idx와 같지 않은 부분이 True인 mask를 tensor.ne(pad_idx)로 계산 (not equal)
        # @@@ 단순 숫자인 pad_idx가 key나 query의 size로 broadcasting 된 후 ne 계산

        mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)  # (n_batch, 1, 1, key_seq_len)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # MHA에서 Q는 (n_batch, n_head, query_seq_len, d_k), K는 (n_batch, n_head, key_seq_len, d_k)인 상태에서
        # Q@K^T / sqrt(d_k) 가 계산되어 (n_batch, n_head, query_seq_len, key_seq_len)인 attention score가 되고
        # 여기서 (n_batch, 1, 1, key_seq_len)인 mask가 broadcasting 되어 
        # (n_batch, n_head, query_seq_len, key_seq_len)인 attention score에 적용
        # ==> query 일반 단어 토큰과 key의 pad 토큰 사이 attention score 계산 값 masking
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        return mask
    def make_tgt_mask_inference(self, tgt, pad_idx):
        pad_mask = self.make_pad_mask_inference(tgt, pad_idx)
        # pad_mask는 (n_batch, 1, 1, tgt_seq_len)꼴
        seq_mask = self.make_subsequent_mask(tgt) 
        # seq_mask는 (tgt_seq_len, tgt_seq_len)꼴

        # pad 부분 mask와 subsequent mask &(and) 연산
        mask = pad_mask & seq_mask
        # @@@ pad_mask와 seq_mask가 (n_batch, 1, tgt_seq_len, tgt_seq_len) 형태로 broadcasting된 뒤 and 연산이 된다
        return mask


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# query pad token과 key의 단어간 attention score 부분은 마스킹하지 않아도 된다
# (결과에서 그 부분을 사용하지 않으므로)
    # def make_pad_mask(self, query, key, pad_idx=2):
    #     # query: (n_batch, query_seq_len)
    #     # key: (n_batch, key_seq_len)
    #     # @@@ decoding block의 cross attention에서는 q와 k가 다르므로 q, k 모두 받는 함수로 생성
    #     # query와 key는 embedding 되기 전의 상태로 각 단어가 vocab안의 해당 인덱스로 저장되어 있음
    #     query_seq_len, key_seq_len = query.size(1), key.size(1)

    #     # pad_idx와 같지 않은 부분이 True인 mask를 tensor.ne(pad_idx)로 계산 (not equal)
    #     # @@@ 단순 숫자인 pad_idx가 key나 query의 size로 broadcasting 된 후 ne 계산

    #     key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)  # (n_batch, 1, 1, key_seq_len)
    #     key_mask = key_mask.repeat(1, 1, query_seq_len, 1)    # (n_batch, 1, query_seq_len, key_seq_len)

    #     query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)  # (n_batch, 1, query_seq_len, 1)
    #     query_mask = query_mask.repeat(1, 1, 1, key_seq_len)  # (n_batch, 1, query_seq_len, key_seq_len)

    #     # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #     # MHA에서 Q는 (n_batch, n_head, query_seq_len, d_k), K는 (n_batch, n_head, key_seq_len, d_k)인 상태에서
    #     # Q@K^T / sqrt(d_k) 가 계산되어 (n_batch, n_head, query_seq_len, key_seq_len)인 attention score가 되고
    #     # 여기서 (n_batch, 1, query_seq_len, key_seq_len)인 mask가 broadcasting 되어 
    #     # (n_batch, n_head, query_seq_len, key_seq_len)인 attention score에 적용
    #     # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    #     mask = key_mask & query_mask
    #     # query와 key 모두 pad가 아닌 부분에서만 attention score가 계산되어야 하므로 &(and) 연산
    #     mask.requires_grad = False
    #     return mask
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@