import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data import MovieLens

import argparse

MOVIE_LENS = "data/ml-1m/ratings.dat"

parser = argparse.ArgumentParser(description='DataModule')
# MLM 관련(Dataset)
parser.add_argument('--max_len', type = int, default = 100)
parser.add_argument('--mask_prob', type = float, default = 0.2)
parser.add_argument('--neg_sample_size',type = int, default = 100)
parser.add_argument('--data_dir',type = str, default = MOVIE_LENS)
# DataLoader 관련
parser.add_argument('--batch_size',type = int, default = 128)
#ml - 1m
parser.add_argument('--vocab_size',type = int, default = 3708)
# default args 설정
default_args = parser.parse_args()

# DataModule 설정
class DataModule(pl.LightningDataModule):
    """
    DataModule : Dataset을 받아와 Dataloader로 반환.
    """
    def __init__(self, args = default_args):
        super(DataModule, self).__init__()
        # seq 최대 길이
        self.max_len = args.max_len
        # masking 기준 확률
        self.mask_prob = args.mask_prob
        # neg sample 수
        self.neg_sample_size = args.neg_sample_size
        # batch_size 설정
        self.batch_size = args.batch_size
        # 데이터 경로
        self.data_dir = args.data_dir
        # 훈련 데이터
        self.movie_train = MovieLens(mode = "Train", max_len = self.max_len, mask_prob = self.mask_prob, data_dir = self.data_dir)
        # vocab size 재할당
        args.vocab_size = self.movie_train.vocab_size
    
    def prepare_data(self):
        """
        데이터 다운로드 진행. 이미 있으니까 pass.
        """
        pass
    def setup(self, stage = None):
        """
        Train/Test 데이터셋 생성.
        """
        # Train 데이터셋 생성
        if stage == "fit" or stage is None:
            """
            Train 데이터셋 불러와야 하지만 vocab_size를 미리 알기 위해서 __init__에서 생성.
            """
            pass
        # Test 데이터셋 생성
        if stage == "test" or stage is None:
            self.movie_test = MovieLens(mode = "Test", max_len = self.max_len, mask_prob = self.mask_prob, data_dir = self.data_dir, \
                                          neg_sample_size = self.neg_sample_size)
    def train_dataloader(self):
        return DataLoader(self.movie_train, batch_size = self.batch_size, shuffle = True)
    
    def val_dataloader(self):
        pass
    
    def test_dataloader(self):
        return DataLoader(self.movie_test, batch_size = self.batch_size)
    # 인스턴스 없이도 확인할 수 있도록
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument('--max_len', type = int, default = 100)
        parser.add_argument('--mask_prob', type = float, default = 0.2)
        parser.add_argument('--neg_sample_size',type = int, default = 100)
        parser.add_argument('--data_dir',type = str, default = MOVIE_LENS)
        # DataLoader 관련
        parser.add_argument('--batch_size',type = int, default = 128)
        #ml - 1m
        parser.add_argument('--vocab_size',type = int, default = 3708)
        return parser