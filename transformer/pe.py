import torch
import torch.nn as nn

import math

# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
# https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch
# https://cpm0722.github.io/pytorch-implementation/transformer
# https://nlp.seas.harvard.edu/2018/04/03/attention.html
# https://nlp.seas.harvard.edu/annotated-transformer/


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout_prob: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)

        self.register_buffer('positional_encodings', get_positional_encoding(d_model, max_len), False)

    def forward(self, x: torch.Tensor):
        # pe = self.positional_encodings[:x.shape[0]].requires_grad_(False)
        # @@@ requires_grad=False인 buffer self.positional_encodings의 slice도 여전히 requires_grad=False이므로 
        # @@@ .requires_grad_(False) 필요 없다

        # self.positional_encodings의 shape는 (1, max_len, d_model)
        # x의 shape는 (n_batch, seq_len, d_model)이므로
        # slicing을 해서 self.positional_encodings의 (1, seq_len, d_model)로 변경
        pe = self.positional_encodings[:, :x.size(1), :]
        x = x + pe
        x = self.dropout(x)
        return x 
    

def get_positional_encoding(d_model: int, max_len: int = 5000):
    # Empty encodings vectors
    encodings = torch.zeros(max_len, d_model)
    # encodings.shape = (max_len, d_model)

    # Position indexes
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    # position.shape = (max_len, 1)

    # 2∗i
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)
    # torch.arange(start, end, step) ==> [start, end) 사이에 start부터 step만큼씩 증가시킨 숫자들 포함
    # two_i.shape = ( celing(d_model/2) )


    # 10000 ^ (-2i/d_model)
    div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))
    # @@@ -를 지수에 곱해 다음 코드에서 / 대신 *를 사용할 수 있게 한다
    # e ^ (2i * -log_e(10000) / d_model) == (e ^(log_e(10000)) ) ^ (-2i / d_model)
    # == 10000 ^ (-2i / d_model) == 1 / (10000 ^ (2i / d_model))

    # PE_{p, 2i} = sin(p / div_term)
    encodings[:, 0::2] = torch.sin(position * div_term)
    # (max_len, 1)과 ( celing(d_model/2) )을 곱하면 
    # broadcasting에 의해 shape가 (max_len, celing(d_model/2)) 이 된다

    # [:, 0::2] : 모든 행, 모든 짝수 열 (start:end:step -> start=0:공백:step=2) 
    # (max_len, d_model)인 encodings의 짝수번째 열들에 대입


    # PE_{p, 2i+1} = cos(p / div_term)
    encodings[:, 1::2] = torch.cos(position * div_term)
    # encodings의 홀수번째 열들에 대입

    # Add batch dimension
    # encodings = encodings.unsqueeze(0).requires_grad_(False)
    encodings = encodings.unsqueeze(0)
    # @@@ encodings는 buffer로 등록되기 댸문에 .requires_grad_(False)안해도 된다

    return encodings
