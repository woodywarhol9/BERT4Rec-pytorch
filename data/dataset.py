import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset
from typing import Dict, Tuple

MOVIE_LENS = "./ml-1m/ratings.dat"
# candidates label
TRUE = 1
FALSE = 0


class MovieLens(Dataset):
    """
    Dataset 생성 및 전처리
    """
    def __init__(self, mode="Train", max_len=100, mask_prob=0.2, data_dir=MOVIE_LENS, neg_sample_size=100):
        self.mode = mode
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.data_dir = data_dir
        self.neg_sample_size = neg_sample_size
        self.user_seq, self.item_seq, self.user2idx, self.item2idx, self.item_size = self._preprocess()
        self.negative_sample = self._popular_sampler(self.item_seq)
        # token 정보
        self.PAD = 0
        self.MASK = len(self.item_seq) + 1

    def _preprocess(self) -> Tuple[pd.DataFrame, pd.Series, Dict[str, int], Dict[str, int], int]:
        """
        데이터를 불러오고 전처리를 수행
        """
        df = pd.read_csv(self.data_dir, sep="::", engine="python")
        df.columns = ["user_id", "item_id", "rating", "timestamp"]
        # 유저, 아이템 인덱싱
        user2idx = {v: k for k, v in enumerate(df['user_id'].unique())}
        item2idx = {v: k + 1 for k, v in enumerate(df['item_id'].unique())}
        item_size = len(item2idx)
        # 유저 - 아이템 매핑
        df['user_id'] = df['user_id'].map(user2idx)
        df['item_id'] = df['item_id'].map(item2idx)
        # 유저 별 시퀀스 생성
        df.sort_values(by='timestamp', inplace=True)
        user_seq = df.groupby(by="user_id")
        user_seq = user_seq.apply(lambda user: list(user["item_id"]))
        
        return user_seq, df.groupby(by="item_id").size(), user2idx, item2idx, item_size

    def _popular_sampler(self, item_seq: pd.Series) -> pd.Index:
        """
        인기 상품 추출
        """
        popular_item = item_seq.sort_values(ascending=False).index
        return popular_item

    def _eval_dataset(self, tokens: list, labels: list) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """
        valid/test dataset 생성
        - LOOV 
        """
        # labl + neg_samples
        candidates = []
        candidates.append(tokens[-1])

        sample_count = 0
        for item in self.negative_sample:
            if sample_count == self.neg_sample_size:
                break
            if item not in set(tokens):
                candidates.append(item)
                sample_count += 1
        # tokens 재정의
        tokens = tokens[:-1] + [self.MASK]
        tokens = tokens[-self.max_len:]  # max_len 까지의 아이템만 활용
        # zero - padding
        pad_len = self.max_len - len(tokens)
        tokens = [self.PAD] * pad_len + tokens
        # candidates labels
        labels = [TRUE] + [FALSE] * self.neg_sample_size

        return torch.LongTensor(tokens), torch.LongTensor(candidates), torch.LongTensor(labels)

    def __len__(self):
        return len(self.user2idx)

    def __getitem__(self, index):

        seq = self.user_seq[index]
        tokens = []
        labels = []

        if self.mode == "Train":
            for item in seq[:-2]:
                prob = np.random.rand()
                # 설정한 mask_prob보다 작을 경우
                if prob < self.mask_prob:
                    prob /= self.mask_prob
                    if prob < 0.8:  # 80%의 아이템은 [MASK]로 변경
                        tokens.append(self.MASK)
                    elif prob < 0.9:  # 10%의 아이템은 랜덤 아이템으로 변경
                        tokens.append(np.random.randint(1, self.item_size + 1))
                    else:  # 10%의 아이템은 그대로 둠
                        tokens.append(item)
                    labels.append(item)  # MLM Label
                # mask prob 이상일 경우 MLM에 사용하지 않음
                else:
                    tokens.append(item)
                    labels.append(self.PAD)  # 활용되지 않으므로 PAD 처리
            # tokens 재정의
            tokens = tokens[-self.max_len:]  
            labels = labels[-self.max_len:]  
            pad_len = self.max_len - len(tokens)
            # zero - padding
            tokens = [self.PAD] * pad_len + tokens
            labels = [self.PAD] * pad_len + labels

            return torch.LongTensor(tokens), torch.LongTensor(labels)

        elif self.mode == "Valid":
            tokens = seq[:-1]
            return self._eval_dataset(tokens, labels)

        elif self.mode == "Test":
            tokens = seq[:]
            return self._eval_dataset(tokens, labels)