import torch
import torch.nn as nn

# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
# https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch
# https://cpm0722.github.io/pytorch-implementation/transformer
# https://nlp.seas.harvard.edu/2018/04/03/attention.html
# https://nlp.seas.harvard.edu/annotated-transformer/

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # .data대신 안전한 .detach()로 변경
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        # x.size() == (n_batch * seq_len, len_vocab)
        # target.size() == (n_batch * seq_len)
        assert x.size(1) == self.size
        true_dist = x.detach().clone()
        # true_dist.size() == (n_batch * seq_len, len_vocab)

        # 전체 길이(vocab length)에서 정답과 패드를 뺸 나머지 self.size - 2 로 
        # smoothing을 나눈값을 채운다
        # ====> 패드 0 + 정답 (confidence == 1-smoothing) + 나머지 전부 smoothing/(self.size - 2) * (self.size - 2) == 1
        # ====> 확률 총합 1
        true_dist.fill_(self.smoothing / (self.size - 2))

        # Tensor.scatter_(dim, index, src, *, reduce=None) → Tensor
        # target.data를 (n_batch * seq_len, 1)로 변경해 true_dist와 동일한 2차원으로 변경
        # true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist.scatter_(1, target.detach().unsqueeze(1), self.confidence)
        # true_dist의 dim = 1 방향(len_vocab)으로 target.data에 적힌 idx(정답 위치)만큼 간 위치에 
        # self.confidence 입력

        # pad 자리는 확률 값 0으로 변경
        true_dist[:, self.padding_idx] = 0

        # 정답이 pad 토큰인 경우 masking
        mask = (target == self.padding_idx)
        # mask.size() == (n_batch * seq_len)

        # 토큰 개수 같은지 확인
        assert mask.size(0) == x.size(0)

        # mask를 (n_batch * seq_len, 1)로 만들어 broadcast 가능하게 만든 뒤 masked_fill_ 사용
        true_dist.masked_fill_(mask.unsqueeze(1), value=0.0)

        # # mask = torch.nonzero(target.data == self.padding_idx)
        # mask = torch.nonzero(target.detach() == self.padding_idx)

        # # 정답이 pad 토큰인 부분의 index를 torch.nonzero가 반환
        # # if mask.dim() > 0: # ??? 정답에 pad 토큰인 부분이 하나도 없어도 mask.dim() == 2?
        # if mask.numel() > 0: 
        #     true_dist.index_fill_(dim=0, index=mask.squeeze(), value=0.0)
        #     # 지정한 dim 방향의 index 위치에 있는 부분의 모든 값을 value로 변경
        #     # ==> 2차원인 true_dist(n_batch * seq_len, len_vocab)의 dim=0방향으로 
        #     # ==> 정답이 pad 토큰인 행을 index로 선택 후 그 행의 모든 값을 0으로 변경
        
        
        self.true_dist = true_dist

        # 예측값 최종 계산에서 log softmax대신 그냥 softmax사용했으므로
        # x.log()입력
        # @@@ nn.KLDivLoss에는 예측은 log softmax값, 정답은 one-hot or softmax or 일반 확률 분포 입력
        return self.criterion(x.log(), true_dist.detach().clone())
        # nn.KLDivLoss(reduction="sum") -> reduction이 sum이므로 
        # n_batch * seq_len개의 토큰 별 KL div 값을 모두 구한 뒤 모든 값을 더해서 하나의 스칼라 값을 반환