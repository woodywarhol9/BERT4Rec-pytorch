import pytorch_lightning as pl

from data import MovieLens

from torch.utils.data import DataLoader
from typing import Optional

MOVIE_LENS = "data/ml-1m/ratings.dat"


class DataModule(pl.LightningDataModule):
    """
    DataModule
    - train/valid/test/predict dataloader 생성
    """
    def __init__(self, args):
        """
        DataModule 초기화
        - Dataset args
        """
        super(DataModule, self).__init__()
        # Dataset 관련 설정
        self.max_len = args.max_len
        self.mask_prob = args.mask_prob
        self.neg_sample_size = args.neg_sample_size
        self.data_dir = args.data_dir
        # DataLoader 관련 설정
        self.pin_memory = args.pin_memory
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        # vocab size 할당
        self.movie_train = MovieLens(
            mode="Train", max_len=self.max_len, mask_prob=self.mask_prob, data_dir=self.data_dir)
        args.item_size = self.movie_train.item_size

    def prepare_data(self):
        """
        Data 다운로드 진행
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Train/Valid/Test 데이터셋 생성
        """
        # Train/Valid 데이터셋 생성
        # args.item size 할당을 위해 train dataset은 __init__에서 정의
        if stage == "fit" or stage is None:
            self.movie_valid = MovieLens(mode="Valid", max_len=self.max_len, mask_prob=self.mask_prob, data_dir=self.data_dir,
                                         neg_sample_size=self.neg_sample_size)
        # Test 데이터셋 생성
        if stage == "test" or stage is None:
            self.movie_test = MovieLens(mode="Test", max_len=self.max_len, mask_prob=self.mask_prob, data_dir=self.data_dir,
                                        neg_sample_size=self.neg_sample_size)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.movie_train, batch_size=self.batch_size, shuffle=True, pin_memory=self.pin_memory, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.movie_valid, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.movie_test, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory, num_workers=self.num_workers)

    def predict_dataloader(self) -> DataLoader:
        pass

    @staticmethod
    def add_to_argparse(parser):
        """
        DataModule 관련 args 정의
        """
        # Dataset 관련
        parser.add_argument('--max_len', type=int, default=100)
        parser.add_argument('--mask_prob', type=float, default=0.2)
        parser.add_argument('--neg_sample_size', type=int, default=100)
        # DataLoader 관련
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--pin_memory', type=bool, default=True)
        parser.add_argument('--num_workers', type=int, default=4)
        # ml - 1m
        parser.add_argument('--item_size', type=int, default=3706)
        parser.add_argument('--data_dir', type=str, default=MOVIE_LENS)

        return parser
