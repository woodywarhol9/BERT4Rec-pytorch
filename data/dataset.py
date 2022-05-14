from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

MOVIE_LENS = "./ml-1m/ratings.dat"
# pad, mask token idx
PAD = 0
MASK = 1
# 정답 label 정보
TRUE = 1
FALSE = 0

class MovieLens(Dataset):
    def __init__(self, mode = "Train", max_len = 100, mask_prob = 0.2, data_dir = MOVIE_LENS, neg_sample_size = 100):
        
        self.mode = mode
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.data_dir = data_dir
        self.neg_sample_size = neg_sample_size
        self.user_seq, self.item_seq, self.user2idx, self.item2idx, self.vocab_size = self.preprocess()
        self.negative_sample = self.popular_sampler(self.item_seq)
        
    def preprocess(self):
        """
        데이터를 불러오고 전처리를 수행.
        """
        df = pd.read_csv(self.data_dir, sep = "::", engine = "python")
        df.columns = ["user_id", "item_id", "rating", "timestamp"]
        # 유저, 아이템 인덱싱
        # 아이템의 idx는 2부터 시작 ( pad -> 0, mask -> 1 )
        user2idx = {v: k for k, v in enumerate(df['user_id'].unique())}
        item2idx = {v: k + 2 for k, v in enumerate(df['item_id'].unique())}
        # vocab_size
        vocab_size = len(item2idx) + 2
        # 유저 - 아이템 매핑
        df['user_id'] = df['user_id'].map(user2idx)
        df['item_id'] = df['item_id'].map(item2idx)
        # 유저 별 시퀀스 생성을 위해 정렬
        df.sort_values(by='timestamp', inplace=True)
        # 시퀀스 데이터 생성을 위해 userId로 그룹화
        # pad, mask 토큰으로 token
        return df.groupby(by='user_id'), df.groupby(by="item_id").size(), user2idx, item2idx, vocab_size
    
    def popular_sampler(self, item_seq):
        """
        인기 상품 추출 
        """
        popular_item = item_seq.sort_values(ascending = False).index
        return popular_item
        
    def __len__(self):
        return len(self.user2idx)
    
    def __getitem__(self, index):
        
        seq = self.user_seq[index]
        
        tokens = []
        labels = []
        # Train dataset 생성
        if self.mode == "Train":   
            for item in seq:
                prob = np.random.random()
                # 설정한 mask_prob보다 작을 경우
                if prob < self.mask_prob:
                    prob /= self.mask_prob  
                    # mask 진행
                    if prob < 0.8: # 80%의 아이템은 [MASK]로 변경한다.
                        # masking
                        tokens.append(MASK) # [MASK] 토큰 할당
                    elif prob < 0.9: # 10%의 아이템은 랜덤한 아이템으로 변경한다
                        tokens.append(np.random.randint(2, self.vocab_size)) # [PAD], [MASK] 제외한 랜덤 아이템 idx로 할당
                    else:
                        tokens.append(item)
                        # 10%의 아이템은 동일하게 둔다.
                    labels.append(item)  # MLM을 위해서 Label을 저장해둔다.
                # mask prob 이상일 경우 그대로 넣어줌.
                else:
                    tokens.append(item) 
                    labels.append(PAD) # MLM에 활용되지 않기 때문에 예측할 필요가 없으므로 Label 0을 할당
                                              
            tokens = tokens[-self.max_len:] # max_len 까지의 아이템만 활용
            labels = labels[-self.max_len:] # max_len 까지의 아이템만 활용
            mask_len = self.max_len - len(tokens) 
            # zero padding, seq 길이가 짧을 경우 zero-padding 진행
            tokens = [PAD for _ in range(mask_len)]  + tokens
            labels = [PAD for _ in range(mask_len)] + labels
            # index 이기 때문에 longTensor로 반환
            return torch.LongTensor(tokens), torch.LongTensor(labels)
        # Test dataset 생성                              
        elif self.mode == "Test":
        # leave one out validation 진행                                
            tokens = seq[:]
            # negative sampling에 사용될 토큰 담기                     
            candidates = []
            candidates.append(seq[-1])
            # 100개의 sample 넣기
            sample_count = 0
            for item in self.negative_sample:
                if sample_count == self.neg_sample_size:
                    break
                # 토큰 추가
                if item not in set(tokens):
                    candidates.append(item)
                    sample_count += 1
            # token 재 할당
            tokens = tokens[:-1] + [MASK]
            tokens = tokens[-self.max_len:] # max_len 까지의 아이템만 활용
            # 얼마나 mask할지 결정
            mask_len = self.max_len - len(tokens)
            # mask 진행
            tokens = [PAD for _ in range(mask_len)] + tokens
            # labels 정보 입력
            labels = [TRUE] + [FALSE] * self.neg_sample_size
            # seq, target + neg sample token, tartget + neg sample
            return torch.LongTensor(tokens), torch.LongTensor(candidates), torch.LongTensor(labels)