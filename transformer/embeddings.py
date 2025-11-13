import torch
import torch.nn as nn

import math

# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
# https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch
# https://cpm0722.github.io/pytorch-implementation/transformer
# https://nlp.seas.harvard.edu/2018/04/03/attention.html
# https://nlp.seas.harvard.edu/annotated-transformer/


class Embeddings(nn.Module):
    def __init__(self, len_vocab, embedding_dim, padding_idx):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=len_vocab, embedding_dim=embedding_dim, padding_idx=padding_idx)
        # nn.Embedding에 padding_idx로 지정된 토큰은
        # 임베딩 후에 embedding_dim개의 모든 엔트리가 0이 된다
        self.embeddding_dim = embedding_dim

    def forward(self, x):
        eb = self.embed(x)
        eb = eb * math.sqrt(self.embeddding_dim)
        return eb
