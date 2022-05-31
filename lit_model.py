import pytorch_lightning as pl
import torch.nn as nn
import torch
from torchmetrics import RetrievalHitRate, RetrievalNormalizedDCG

from model import BERT

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
        self.vocab_size = args.item_size + 2
        self.initializer_range = args.initializer_range

        self.neg_sample_size = args.neg_sample_size
        # Optimizer 관련 파라미터
        self.weight_decay = args.weight_decay
        self.decay_step = args.decay_step
        self.gamma = args.gamma
        # BERT 모델 생성
        self.model = BERT(
            vocab_size=self.vocab_size,
            max_len=self.max_len,
            hidden_dim=self.hidden_dim,
            layer_num=self.layer_num,
            head_num=self.head_num,
            dropout_rate=self.dropout_rate,
            dropout_rate_attn=self.dropout_rate_attn,
            initializer_range=self.initializer_range
        )
        # 토큰 예측
        # Mask의 예측이므로 args.item_size + 1
        self.out = nn.Linear(self.hidden_dim, args.item_size + 1)
        # criterion 생성
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        # HR@10, NDGC@10
        # val, test 각각 선언.
        self.batch_size = args.batch_size

        self.HR = RetrievalHitRate(k=10)
        self.NDCG = RetrievalNormalizedDCG(k=10)

    def forward(self, x):
        """
        seq로 mask 예측하기.
        """
        logits = self.model(x)
        preds = self.out(logits)
        preds = preds[:, -1, :]  # 마지막 Mask 예측
        # top10 상품 예측
        indexes, _ = torch.topk(preds, 10)

        return indexes

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
        # labels : [batch_size, max_len]
        loss = self.criterion(preds.transpose(1, 2), labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        검증 : Mask Prediction 수행
        """
        seq, candidates, labels = batch
        logits = self.model(seq)
        preds = self.out(logits)

        preds = preds[:, -1, :]  # MASK 부분만 뽑기
        targets = candidates[:, 0]  # 첫 Candidates가 Label
        # preds : [bach_size, vocab_size]
        # loss 계산
        # candidates[:,0] : [batch_size] -> 각 배치의 0번 candidates 정보
        loss = self.criterion(preds, targets)
        # neg_sample index 선택
        # recs : [batch_size, neg_sample + 1]
        recs = torch.gather(preds, 1, candidates)
        # HR, NDCG 구하기
        steps = batch_idx * self.batch_size
        indexes = torch.arange(steps, steps + seq.size(0), dtype=torch.long).unsqueeze(1).repeat(1, 101)
        # HR@10, NDGC@10
        hr = self.HR(recs, labels, indexes)
        ndcg = self.NDCG(recs, labels, indexes)
        # 기록
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("HR_val", hr, on_step = False, on_epoch=True, prog_bar = True, logger = True)
        self.log("NDCG_val", ndcg, on_step = False, on_epoch=True, prog_bar = True, logger = True)

    def test_step(self, batch, batch_idx):
        """
        테스트 : Mask Prediction 수행
        """
        seq, candidates, labels = batch
        logits = self.model(seq)
        preds = self.out(logits)

        preds = preds[:, -1, :]  # MASK 부분만 뽑기
        targets = candidates[:, 0]  # 첫 Candidates가 Label
        # preds : [bach_size, vocab_size]
        # loss 계산
        # candidates[:,0] : [batch_size] -> 각 배치의 0번 candidates 정보
        loss = self.criterion(preds, targets)
        # neg_sample index 선택
        # recs : [batch_size, neg_sample + 1]
        recs = torch.gather(preds, 1, candidates)
        # HR, NDCG 구하기
        steps = batch_idx * self.batch_size
        indexes = torch.arange(steps, steps + seq.size(0), dtype=torch.long).unsqueeze(1).repeat(1, 101)
        # HR, NDCG 구하기
        hr = self.HR(recs, labels, indexes)
        ndcg = self.NDCG(recs, labels, indexes)
        # 기록
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("HR_test", hr, on_step = False, on_epoch =True, prog_bar = True, logger = True)
        self.log("NDCG_test", ndcg, on_step = False, on_epoch =True, prog_bar = True, logger = True)

    def configure_optimizers(self):
        """
        Optimizer, Scheduler 초기화.
        """
        no_decay = ['bias', 'LayerNorm.weight']
        #No decay bias, LayerNorm.weight
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                                        ]
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.decay_step, gamma=self.gamma
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    # 인스턴스 없이도 확인할 수 있도록 static method로 설정
    @staticmethod
    def add_to_argparse(parser):
        """
        Training 관련 args 입력.
        """
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        # BERT 모델 관련
        parser.add_argument("--hidden_dim", type=int, default=256)
        parser.add_argument("--layer_num", type=int, default=2)
        parser.add_argument("--head_num", type=int, default=4)
        parser.add_argument("--dropout_rate", type=float, default=0.1)
        parser.add_argument("--dropout_rate_attn", type=float, default=0.1)
        parser.add_argument("--initializer_range", type=float, default=0.02)
        # Optimizer 관련
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--decay_step", type=int, default=25)
        parser.add_argument("--gamma", type=float, default=0.1)

        return parser