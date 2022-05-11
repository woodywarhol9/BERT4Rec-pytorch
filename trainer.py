import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from .data import MovieLens
from .model import BERT

import argparse

MOVIE_LENS = "./ml-1m/ratings.dat"

parser = argparse.ArgumentParser(description='BERT4REC')
# 입력받을 인자값 설정
# 학습 관련
parser.add_argument('--epoch',          type=int,   default=100)
parser.add_argument('--batch_size',     type=int,   default=128)
parser.add_argument('--lr',     type=float, default=1e-3)
# BERT 모델 관련
parser.add_argument('--max_len',          type=int,   default=100)
parser.add_argument('--hidden_dim',     type=int,   default=256)
parser.add_argument('--layer_num',     type=float, default=2)
parser.add_argument('--head_num',          type=int,   default=2)
parser.add_argument('--dropout_ratio',     type=int,   default=0.1)
parser.add_argument('--dropout_ratio_attn',     type=float, default=0.1)
# MLM 관련(Dataset)
parser.add_argument('--mask_prob',     type=int,   default=0.2)
parser.add_argument('--data_dir',     type=str, default = MOVIE_LENS)
# args 에 위의 내용 저장
default_args = parser.parse_args()

class BERT4REC(pl.LightningModule):
    def __init__(self, args = default_args):
        """
        모델 및 데이터셋 불러오기.
        """
        super(BERT4REC, self).__init__()
        # 학습 관련 파라미터
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.lr = args.lr
        # BERT 관련 파라미터
        self.max_len = args.max_len
        self.hidden_dim = args.hidden_dim
        self.layer_num = args.layer_num
        self.head_num = args.head_num
        self.dropout_ratio = args.dropout_ratio
        self.dropout_ratio_attn = args.dropout_ratio_attn
        # Dataset 관련 파라미터
        self.mask_prob = args.mask_prob
        self.data_dir = args.data_dir
        
        self.dataset = MovieLens(max_len = self.max_len, mask_prob = self.mask_prob, data_dir = self.data_dir)
        self.vocab_size = dataset.vocab_size
        self.model = BERT(vocab_size = self.vocab_size, max_len = self.max_len, hidden_dim = self.hidden_dim, layer_num = self.layer_num, \
                         head_num = self.head_num, dropout_ratio = self.dropout_ratio, dropout_ratio_attn = self.dropout_ratio_attn)

    def forward(self, x):


    def training_step(self, batch, batch_idx):


    def configure_optimizers(self):