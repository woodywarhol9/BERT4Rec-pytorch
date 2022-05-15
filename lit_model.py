import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchmetrics.functional import retrieval_normalized_dcg, retrieval_hit_rate

from model import BERT

import argparse

parser = argparse.ArgumentParser(description='LitModule')
# 입력받을 인자값 설정
parser.add_argument('--learning_rate', type = float, default = 1e-3)
# BERT 모델 관련
parser.add_argument('--max_len', type = int, default = 100)
parser.add_argument('--hidden_dim', type = int, default = 256)
parser.add_argument('--layer_num', type = int, default = 2)
parser.add_argument('--head_num', type = int, default = 2)
parser.add_argument('--dropout_rate', type = float, default = 0.1)
parser.add_argument('--dropout_rate_attn', type = float, default = 0.1)
# vocab_size
parser.add_argument('--vocab_size', type = int, default = 3708)
# args 에 위의 내용 저장
default_args = parser.parse_args()


class BERT4REC(pl.LightningModule):
    def __init__(self, args):
        """
        모델 및 데이터셋 불러오기.
        """
        super(BERT4REC, self).__init__()
        # 학습 관련 파라미터
        self.learning_rate = args.learning_rate
        # BERT 관련 파라미터
        self.max_len = args.max_len
        self.hidden_dim = args.hidden_dim
        self.layer_num = args.layer_num
        self.head_num = args.head_num
        self.dropout_rate = args.dropout_rate
        self.dropout_rate_attn = args.dropout_rate_attn
        self.vocab_size = args.vocab_size

        self.model = BERT(vocab_size = self.vocab_size, max_len = self.max_len, hidden_dim = self.hidden_dim, layer_num = self.layer_num, \
                         head_num = self.head_num, dropout_rate = self.dropout_rate, dropout_rate_attn = self.dropout_rate_attn)
        # 예측
        self.out = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, x):
        """
        seq로 mask 예측하기.
        """
        logits = self.model(x)
        preds = self.out(logits)
        preds = preds[:, -1, :] # 마지막 Mask 예측
        # top10 상품 예측
        _ , recs = torch.topk(preds, 10)

        return recs

    def training_step(self, batch, batch_idx):
        """
        훈련 : BERT MLM 수행
        """
        seq, labels = batch
        # logits : [batch_size, max_len, hidden_dim]
        logits = self.model(seq)
        # preds : [batch_size, max_len, vocab_size]
        preds = self.out(logits)
        # nn.CrossEntropyLoss : log softmax + NLL loss
        # pad는 예측에 제외되므로 ignore_index에 포함
        print("TRAIN",preds, labels)
        loss = nn.CrossEntropyLoss(preds, labels, ignore_index = 0)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        검증 : Mask Prediction 수행
        """
        seq, candidates, labels = batch
        logits = self.model(seq)
        preds = self.out(logits)
        preds = preds[:,-1,:] # MASK 부분만 뽑기
        # loss 계산
        loss = nn.CrossEntropyLoss(preds, candidates[:, 0], ignore_index = 0)
        # 해당 상품 index 선택
        recs = torch.take(preds, candidates)
        # HR, NDCG 구하기
        hr = retrieval_hit_rate(recs, labels, k = 10)
        ndcg = retrieval_normalized_dcg(recs, labels, k = 10)
        # 기록
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("HR_val", hr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("NDCG_val", ndcg, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        테스트 : Mask Prediction 수행
        """
        seq, candidates, labels = batch

        logits = self.model(seq)
        preds = self.out(logits)
        preds = preds[:,-1,:] # MASK 부분만 뽑기
        # loss 계산
        loss = nn.CrossEntropyLoss(preds, candidates[:, 0], ignore_index = 0)
        # 해당 상품 index 선택
        recs = torch.take(preds, candidates)
        # HR, NDCG 구하기
        hr = retrieval_hit_rate(recs, labels, k = 10)
        ndcg = retrieval_normalized_dcg(recs, labels, k = 10)
        # 기록
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("HR_val", hr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("NDCG_val", ndcg, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay = 0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                        T_0=10,
                                                                        T_mult=1,
                                                                        eta_min=1e-3,
                                                                        last_epoch=-1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    # 인스턴스 없이도 확인할 수 있도록 static method로 설정
    @staticmethod
    def add_to_argparse(parser):
        # 학습 관련
        parser.add_argument('--learning_rate', type = float, default = 1e-3)
        # BERT 모델 관련
        parser.add_argument('--hidden_dim', type = int, default = 256)
        parser.add_argument('--layer_num', type = int, default = 2)
        parser.add_argument('--head_num', type = int, default = 2)
        parser.add_argument('--dropout_rate', type = float, default = 0.1)
        parser.add_argument('--dropout_rate_attn', type = float, default = 0.1)

        return parser