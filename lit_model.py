import pytorch_lightning as pl
import torch.nn as nn
import torch
import numpy as np

from model import BERT

from torchmetrics import RetrievalHitRate, RetrievalNormalizedDCG


class BERT4REC(pl.LightningModule):
    """
    LightninModule
    - train, valid, test, predict step 정의
    - optimizer, scheduler 정의
    - Module 관련 args 정의
    """
    def __init__(self, args):
        """
        LightningModule 초기화
        - model 생성
        - criterion, metrics 정의
        - train/validate-test step 정의
        - prediction(inference) 정의
        """
        super(BERT4REC, self).__init__()
        # 학습 관련 파라미터
        self.learning_rate = args.learning_rate
        # BERT 관련 파라미터
        self.max_len = args.max_len
        self.hidden_dim = args.hidden_dim
        self.encoder_num = args.encoder_num
        self.head_num = args.head_num
        self.dropout_rate = args.dropout_rate
        self.dropout_rate_attn = args.dropout_rate_attn
        self.vocab_size = args.item_size + 2
        self.initializer_range = args.initializer_range
        # Optimizer 관련 파라미터
        self.weight_decay = args.weight_decay
        self.decay_step = args.decay_step
        self.gamma = args.gamma
        # BERT 생성
        self.model = BERT(
            vocab_size=self.vocab_size,
            max_len=self.max_len,
            hidden_dim=self.hidden_dim,
            encoder_num=self.encoder_num,
            head_num=self.head_num,
            dropout_rate=self.dropout_rate,
            dropout_rate_attn=self.dropout_rate_attn,
            initializer_range=self.initializer_range
        )
        self.out = nn.Linear(self.hidden_dim, args.item_size + 1)  # Mask 예측 :  1 ~ args.item_size + 1
        self.batch_size = args.batch_size  # steps 계산에 활용
        # criterion, metircs
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.HR = RetrievalHitRate(k=10)  # HR@10
        self.NDCG = RetrievalNormalizedDCG(k=10)  # NDCG@10

    def forward(self, x):
        """
        debugging에 사용
        """
        pass

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        trainining 진행
        - loss를 return해야 back-prop 가능
        """
        seq, labels = batch
        logits = self.model(seq)  # logits : [batch_size, max_len, hidden_dim]
        preds = self.out(logits)  # preds : [batch_size, max_len, vocab_size]

        loss = self.criterion(preds.transpose(1, 2), labels)
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """
        validation 진행
        """
        seq, candidates, labels = batch
        logits = self.model(seq)
        preds = self.out(logits)

        preds = preds[:, -1, :]  # MASK 부분만 뽑기, preds : [bach_size, vocab_size]
        targets = candidates[:, 0]  # 각 batch의 0 idx Candidates가 Label
        loss = self.criterion(preds, targets)

        recs = torch.gather(preds, 1, candidates)  # recs : [batch_size, neg_sample + 1]
        # HR, NDCG 구하기
        steps = batch_idx * self.batch_size
        indexes = torch.arange(steps, steps + seq.size(0), dtype=torch.long).unsqueeze(1).repeat(1, 101)
        hr = self.HR(recs, labels, indexes)  # dim recs = labels = indexes
        ndcg = self.NDCG(recs, labels, indexes)
        # on_step = False, on_epoch = True : validation_step이 모두 끝난 후에 log
        self.log("val_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log("HR_val", hr, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log("NDCG_val", ndcg, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """
        test 진행
        """
        seq, candidates, labels = batch
        logits = self.model(seq)
        preds = self.out(logits)

        preds = preds[:, -1, :]  # MASK 부분만 뽑기, preds : [bach_size, vocab_size]
        targets = candidates[:, 0]  # 각 batch의 0 idx Candidates가 Label
        loss = self.criterion(preds, targets)

        recs = torch.gather(preds, 1, candidates)  # recs : [batch_size, neg_sample + 1]
        # HR, NDCG 구하기
        steps = batch_idx * self.batch_size
        indexes = torch.arange(steps, steps + seq.size(0), dtype=torch.long).unsqueeze(1).repeat(1, 101)
        hr = self.HR(recs, labels, indexes)  # dim recs = labels = indexes
        ndcg = self.NDCG(recs, labels, indexes)
        # on_step = False, on_epoch = True : test_step이 모두 끝난 후에 log
        self.log("test_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log("HR_test", hr, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log("NDCG_test", ndcg, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, batch: torch.Tensor, batch_idx : int, dataloader_idx=0) -> np.array:
        """
        predict 진행
        """
        seq = batch
        logits = self.model(seq)
        preds = self.out(logits)

        preds = preds[:, -1, :]  # MASK 부분만 뽑기, preds : [bach_size, vocab_size]
        indexes, _ = torch.topk(preds, 10)

        return indexes.cpu().numpy()

    def configure_optimizers(self):
        """
        Optimizer, Scheduler 초기화
        - return optimizer, scheduler, monitor
        """
        # No decay bias, LayerNorm.weight
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.Adam(
            optimizer_grouped_parameters, lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.decay_step, gamma=self.gamma
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    @staticmethod
    def add_to_argparse(parser):
        """
        litmodule 관련 args 정의
        """
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        # BERT 모델 관련
        parser.add_argument("--hidden_dim", type=int, default=256)
        parser.add_argument("--encoder_num", type=int, default=2)
        parser.add_argument("--head_num", type=int, default=4)
        parser.add_argument("--dropout_rate", type=float, default=0.1)
        parser.add_argument("--dropout_rate_attn", type=float, default=0.1)
        parser.add_argument("--initializer_range", type=float, default=0.02)
        # Optimizer 관련
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--decay_step", type=int, default=25)
        parser.add_argument("--gamma", type=float, default=0.1)

        return parser
