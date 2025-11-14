import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
# https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch
# https://cpm0722.github.io/pytorch-implementation/transformer
# https://nlp.seas.harvard.edu/2018/04/03/attention.html
# https://nlp.seas.harvard.edu/annotated-transformer/

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# class ScaledDotPruductAttention(nn.Module):
#     def __init__(self, in_feat_num, out_feat_num, qk_feat_num, v_feat_num):
#         super().__init__()
#         # self.in_feat_num = in_feat_num
#         # self.out_feat_num = out_feat_num
#         self.qk_feat_num = qk_feat_num
#         # self.v_feat_num = v_feat_num
#         self.make_Q = nn.Linear(in_features=in_feat_num, out_features=qk_feat_num)
#         self.make_K = nn.Linear(in_features=in_feat_num, out_features=qk_feat_num)
#         self.make_V = nn.Linear(in_features=in_feat_num, out_features=v_feat_num)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, mask=None):
#         Q = self.make_Q(x)
#         K = self.make_K(x)
#         V = self.make_V(x)
#         K_T = K.transpose(-2, -1)

#         attend = (Q @ K_T) / np.sqrt(self.qk_feat_num)
#         # Q@K^T와 scaling

#         # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#         # if mask: 를 쓰면 mask가 None이 아닐 때
#         # RuntimeError: Boolean value of Tensor with more than one value is ambiguous 발생
#         if mask is not None:
#             # attend = attend.masked_fill(mask=mask, value=float("-inf"))
#             attend.masked_fill_(mask=mask, value=float("-inf"))
#             # 양의 무한대는 float("inf"), 음의 무한대는 float("-inf")
#             # torch.Tensor.masked_fill 함수에는 원본 tensor와 동일 size를 가지는 mask입력 (mask는 0, 1 두가지 값만 존재)
#             # 원본 tensor의 각 entry는 mask의 동일 위치 값이 1이면 그대로 두고
#             # 0이면 위의 value에 입력한 값으로 변경한다
#         # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#         attend = self.softmax(attend)
#         out = attend @ V

#         return out
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# 각 ScaledDotPruductAttention 안에서 W_Q, W_K, W_V를 가지고 Q,K,V를 생성하는 경우
# Multi-head attention의 head 개수 h * 각 head별 W_Q, W_K, W_V 3개 해서 3h 개의 weight matrix를 생성해야 한다
# ex: h=8, d_model == 512 이고 d_q_h == d_k_h == d_v_h == 64인 경우
# 512x64인 weight matrix가 8*3 24개 필요
# => 하나의 512차원 input vector가 24개의 512x64 weight matrix와 24번 계산해서 24개의 64차원 vector 생성한다
# ==> 이는 weight matrix들을 열 방향으로 h=8개를 다 붙인 512x(64*8)차원 weight matrix에 input vector를 계산한 후
# (64*8)차원 결과를 64차원 8개로 쪼개주는 방식으로 변경하는게 더 좋다
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


class ScaledDotPruductAttention(nn.Module):
    def __init__(self, scaling_factor):
        super().__init__()
        self.scaling_factor = scaling_factor
        # self.scaling_factor = sqrt(d_k)

    def forward(self, Q, K, V, mask=None):
        K_T = K.transpose(-2, -1)

        attend = (Q @ K_T) / self.scaling_factor
        # Q@K^T와 scaling

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # if mask: 를 쓰면 mask가 None이 아닐 때
        # RuntimeError: Boolean value of Tensor with more than one value is ambiguous 발생
        if mask is not None:
            # attend = attend.masked_fill(mask=mask, value=float("-inf"))
            attend.masked_fill_(mask=(mask == 0), value=float("-inf"))
            # @@@@ attend의 (seq_len, seq_len)행렬에서 pad token과 다른 모든 단어 사이의 score가 계산된 행의 경우
            # @@@@ 행의 모든 엔트리가 float("-inf")로 변환되면 softmax 계산시 0/0이 되어 NaN이 나오는 문제가 있다
            # attend.masked_fill_(mask=(mask == 0), value=float("-1e20"))
            # @@@@ ==> 위와 같이 -1e20과 같이 매우 작은 숫자로 변경하거나 softmax 계산후 torch.where로 NaN 부분 0으로 변경 필요
                # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                # 일반적인 구현 방식들처럼 굳이 query부분 mask를 할 필요는 없다 
                # --> 어차피 query pad token과 key의 모든 단어간의 attention score는 결국 쓰이지 않는다
                # --> query 부분 masking을 하지 않으면 행의 모든 엔트리가 float("-inf")인 경우가 없으므로 softmax계산에 문제 없다
                # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # 양의 무한대는 float("inf"), 음의 무한대는 float("-inf")
            # torch.Tensor.masked_fill 함수에는 원본 tensor와 동일 size를 가지는 mask입력 (mask는 0, 1 두가지 값만 존재)
            # 원본 tensor의 각 entry는 mask의 동일 위치 값이 False이면 그대로 두고
            # True이면 위의 value에 입력한 값으로 변경한다

            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # 들어오는 mask는 pad가 아닌 부분이 True로 되어 있으므로 
            # pad 부분이 True가 되도록 mask == 0을 한 후 입력해야 한다.
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        attend = F.softmax(input=attend, dim=-1)
        out = attend @ V

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, head_num, drop_rate):
        # EncodingBlock에서 input x와 MultiHeadAttention(x)을 더하는 residual connection이 존재
        # => MultiHeadAttention(x)은 x와 동일 차원이어야 한다
        # decoder안에서는 key와 value가 encoder에서 올 수도 있으므로
        # Q와 K의 차원이 다를 수 있다
        # 그러나 각 head로 쪼개질 때 q_h와 k_h는 동일차원으로 맞추어야 inner product 가능
        # ==> q_h, k_h, v_h 모두 self.head_dim으로 통일
        super().__init__()
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.head_num = head_num
        self.head_dim = q_dim // head_num
        # q_dim을 head_num으로 나누는 정수값이 없는 경우
        # multi-head attention 결과를 concat하였을 때 input인 q_dim과 달라질 수 있으나
        # concat한 결과를 FC layer를 한번 거쳐서 q_dim으로 맞춰준다

        self.make_Q = nn.Linear(in_features=q_dim, out_features=self.head_num * self.head_dim)
        self.make_K = nn.Linear(in_features=k_dim, out_features=self.head_num * self.head_dim)
        self.make_V = nn.Linear(in_features=v_dim, out_features=self.head_num * self.head_dim)
        # 각 head가 (b, seq_length, d_model)을 받아 (b, seq_length, self.head_dim)인 q_h, k_h, v_h를 각각 생성하는 대신
        # 한꺼번에 계산해 (b, seq_length, self.head_num * self.head_dim)를 얻은 후
        # 이를 view로 (b, seq_length, self.head_num, self.head_dim) 변경하고 transpose로
        # 최종 (b, self.head_num, seq_length, self.head_dim)로 변경

        # self.SDPAttention = ScaledDotPruductAttention(scaling_factor=np.sqrt(self.head_dim))
        # Q_h, K_h의 각 entry의 평균이 0, 분산은 1이면
        # Q_h @ K_h^T의 (i,k) entry는 ∑_(j=1)^(d_k) (q_ij)*(k_kj) 이므로 분산이 d_k이 된다
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # X,Y가 서로 독립일 때,
        # V(XY) = V(X)V(Y) + V(X)[E(Y)^2] + V(Y)[E(X)^2]
        # V(X+Y) = V(X) + V(Y)
        # ===> 평균이 0이고 분산은 1인 RV q_ij, k_kj들이 ∑_(j=1)^(d_k) (q_ij)*(k_kj) 형태로 되어 있으면
        # (q_ij)*(k_kj)의 분산은 1 (평균이 0이므로 V(XY) = V(X)V(Y) = 1)
        # V[∑_(j=1)^(d_k) (분산 1인 변수들)] = d_k * 1 = d_k
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # => softmax에 입력하기 전에 분산을 1로 맞춰주기 위해서 sqrt(d_k)로 나눠준다 <= V(aX) = a^2 * V(X) ===> V(X/a) = V(X) / a^2
        #   scaling이 없고 d_k값이 크면 분산이 너무 커져서 soft max에 입력되는 값들이 전부 너무 커진다
        #   그리고 softmax의 입력값이 큰 부분의 gradient는 매우 작다 ==> 학습이 잘 안된다)

        # self.SDPAttention = ScaledDotPruductAttention(scaling_factor=np.sqrt(self.head_num * self.head_dim))

        self.make_O = nn.Linear(in_features=self.head_num * self.head_dim, out_features=q_dim)
        # self.head_dim = q_dim // head_num 이므로 나머지가 0이 아닐 경우 q_dim > self.head_num * self.head_dim

        self.dropout = nn.Dropout(p=drop_rate)
        # 논문과 동일하게 output에 dropout 적용
        # layernorm(sublayer_input + dropout(sublayer(sublayer_input)))
        # self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.make_Q.weight)
        self.make_Q.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.make_K.weight)
        self.make_K.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.make_V.weight)
        self.make_V.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.make_O.weight)
        self.make_O.bias.data.fill_(0)

    def SDPA(self, Q, K, V, mask):

        K_T = K.transpose(-2, -1)

        attend = (Q @ K_T) / np.sqrt(self.head_dim)
        # Q@K^T와 scaling

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # if mask: 를 쓰면 mask가 None이 아닐 때
        # RuntimeError: Boolean value of Tensor with more than one value is ambiguous 발생
        if mask is not None:
            # attend = attend.masked_fill(mask=mask, value=float("-inf"))
            attend.masked_fill_(mask=(mask == 0), value=float("-inf"))
            # @@@@ attend의 (seq_len, seq_len)행렬에서 pad token과 다른 모든 단어 사이의 score가 계산된 행의 경우
            # @@@@ 행의 모든 엔트리가 float("-inf")로 변환되면 softmax 계산시 0/0이 되어 NaN이 나오는 문제가 있다
            # attend.masked_fill_(mask=(mask == 0), value=float("-1e20"))
            # @@@@ ==> 위와 같이 -1e20과 같이 매우 작은 숫자로 변경하거나 softmax 계산후 torch.where로 NaN 부분 0으로 변경 필요
                # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                # 일반적인 구현 방식들처럼 굳이 query부분 mask를 할 필요는 없다 
                # --> 어차피 query pad token과 key의 모든 단어간의 attention score는 결국 쓰이지 않는다
                # --> query 부분 masking을 하지 않으면 행의 모든 엔트리가 float("-inf")인 경우가 없으므로 softmax계산에 문제 없다
                # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # 양의 무한대는 float("inf"), 음의 무한대는 float("-inf")
            # torch.Tensor.masked_fill 함수에는 원본 tensor와 동일 size를 가지는 mask입력 (mask는 0, 1 두가지 값만 존재)
            # 원본 tensor의 각 entry는 mask의 동일 위치 값이 False이면 그대로 두고
            # True이면 위의 value에 입력한 값으로 변경한다

            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # 들어오는 mask는 pad가 아닌 부분이 True로 되어 있으므로 
            # pad 부분이 True가 되도록 mask == 0을 한 후 입력해야 한다.
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        attend = F.softmax(input=attend, dim=-1)


        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # The Annotated Transformer 구현과 동일하게 attetion 결과에 dropout 적용
            # ==> attention이 특정 key에 과도하게 치우치지 않고 더 일반화된 패턴을 학습할 수 있도록 돕는다
        attend = self.dropout(attend)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        out = attend @ V

        return out

    def forward(self, Q, K, V, mask):
        b = Q.size(0)
        q_len = Q.size(1)
        k_len = K.size(1)
        v_len = V.size(1)

        Q_ = self.make_Q(Q)
        K_ = self.make_K(K)
        V_ = self.make_V(V)
        # (b, seq_length, self.head_num * self.head_dim)
        # Q는 decoder, K,V는 encoder에서 오는 경우 seq_length가 다를 수 있다

        Q_ = Q_.view(b, q_len, self.head_num, self.head_dim)
        K_ = K_.view(b, k_len, self.head_num, self.head_dim)
        V_ = V_.view(b, v_len, self.head_num, self.head_dim)
        # (b, seq_length, self.head_num, self.head_dim)
        Q_ = Q_.permute(0, 2, 1, 3)
        K_ = K_.permute(0, 2, 1, 3)
        V_ = V_.permute(0, 2, 1, 3)
        # (b, self.head_num, seq_length, self.head_dim)

        out = self.SDPA(Q_, K_, V_, mask)
        # 4차원 tensor끼리 행렬곱은 앞 두차원(batch, head) 번호가 같은 2차원 행렬끼리의 곱
        # ex: Q = (b=16, h=8, Q_seq_length=30, h_dim=64)과 K^T = (b=16, h=8, h_dim=64, K_seq_length=50)를 곱하면 (Q @ K^T)
        #     같은 배치, 같은 head에 있는 30x64, 64x50 행렬이 곱해진다
        # out.size() = (b, self.head_num, Q_seq_length, d_v==self.head_dim)
        out = out.permute(0, 2, 1, 3)
        # (b, Q_seq_length, self.head_num, self.head_dim)로 다시 변경
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # https://medium.com/@serotoninpm/pytorch-%EB%AC%B8%EB%B2%95-view-reshape-transpose-permute-contiguous-f779711cbc6f
        # view 함수는 contiguous한 tensor에만 사용할 수 있다
        # 따라서 permute나 transpose후에 view를 사용하려면 .contiguous()를 사용해야한다
        out = out.contiguous().view(b, q_len, -1)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # $$ reshape() == contiguous().view()를 사용해도 된다
        # $$ out = out.reshape(b, q_len, -1)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # (b, Q_seq_length, self.head_num * self.head_dim)
        # => multi-head결과를 concat
        out = self.make_O(out)
        # 마지막 linear를 거쳐
        # (b, Q_seq_length, self.q_dim) 출력
        out = self.dropout(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, in_dim, h_dim, drop_rate):
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.drop_rate = drop_rate

        self.fc1 = nn.Linear(in_features=in_dim, out_features=h_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=h_dim, out_features=in_dim)

        self.dropout = nn.Dropout(p=drop_rate)
        # 논문과 동일하게 output에 dropout 적용
        # self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # The Annotated Transformer 구현과 동일하게 중간에 dropout 추가
        x = self.dropout(x)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        x = self.fc2(x)

        x = self.dropout(x)

        return x


class EncodingBlock(nn.Module):
    def __init__(self, q_dim, h_dim, head_num, drop_rate):
        super().__init__()
        self.MHA = MultiHeadAttention(
            q_dim=q_dim, k_dim=q_dim, v_dim=q_dim, head_num=head_num, drop_rate=drop_rate
        )
        # self attention
        # (b, seq_len, q_dim) => (b, seq_len, q_dim)

        self.FF = FeedForward(in_dim=q_dim, h_dim=h_dim, drop_rate=drop_rate)
        self.LN1 = nn.LayerNorm(q_dim)
        self.LN2 = nn.LayerNorm(q_dim)

    def forward(self, Q, mask):
        # @@@ pad 부분과 실제 단어 사이에 attention을 0으로 mask하는 mask 필수?
        out1 = self.LN1(Q + self.MHA(Q, Q, Q, mask))
        # self attention이므로 input 하나로 Q,K,V 전부 생성
        # residual connection 연결 후 layer norm 실행

        out2 = self.LN2(out1 + self.FF(out1))
        # out2는 (b, seq_len, d_model==q_dim) 형태

        # TODO:post-norm을 pre-norm으로 바꿔보기
        return out2


class DecodingBlock(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, h_dim, head_num, drop_rate):
        super().__init__()
        self.MMHA = MultiHeadAttention(
            q_dim=q_dim, k_dim=q_dim, v_dim=q_dim, head_num=head_num, drop_rate=drop_rate
        )
        # Masked self-attention

        self.MHA = MultiHeadAttention(
            q_dim=q_dim, k_dim=k_dim, v_dim=v_dim, head_num=head_num, drop_rate=drop_rate
        )
        # Encoder-Decoder attention
        # Q는 위의 MMHA의 output, K,V는 encoder output

        self.FF = FeedForward(in_dim=q_dim, h_dim=h_dim, drop_rate=drop_rate)

        self.LN1 = nn.LayerNorm(q_dim)
        self.LN2 = nn.LayerNorm(q_dim)
        self.LN3 = nn.LayerNorm(q_dim)

    def forward(self, Q, K, V, tgt_mask, src_mask):
        out1 = self.LN1(Q + self.MMHA(Q, Q, Q, tgt_mask))
        # Masked self-attention이므로 query, key, value 모두 Q

        out2 = self.LN2(out1 + self.MHA(out1, K, V, src_mask))
        # encoder-decoder attention은 key와 value는 encoder 출력을 입력받아 사용

        out3 = self.LN3(out2 + self.FF(out2))

        # TODO:post-norm을 pre-norm으로 바꿔보기
        return out3