import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data import MovieLens

MOVIE_LENS = "data/ml-1m/ratings.dat"

# DataModule 설정


class DataModule(pl.LightningDataModule):
    """
    DataModule : Dataset을 받아와 Dataloader로 반환.
    """

    def __init__(self, args):
        super(DataModule, self).__init__()
        # seq 최대 길이
        self.max_len = args.max_len
        # masking 기준 확률
        self.mask_prob = args.mask_prob
        # neg sample 수
        self.neg_sample_size = args.neg_sample_size
        # 데이터 경로
        self.data_dir = args.data_dir
        # batch_size 설정
        self.batch_size = args.batch_size
        # DataLoader 관련 설정
        self.pin_memory = args.pin_memory
        self.num_workers = args.num_workers
        # 훈련 데이터
        self.movie_train = MovieLens(
            mode="Train", max_len=self.max_len, mask_prob=self.mask_prob, data_dir=self.data_dir)
        # vocab size 재할당
        args.item_size = self.movie_train.item_size

    def prepare_data(self):
        """
        데이터 다운로드 진행. 이미 있으니까 pass.
        """
        pass

    def setup(self, stage=None):
        """
        Train/Valid/Test 데이터셋 생성.
        """
        # Train/Valid 데이터셋 생성
        if stage == "fit" or stage is None:
            # train data
            # vocab_size 얻기 위해서 미리 선언.
            # valid data
            self.movie_valid = MovieLens(mode="Valid", max_len=self.max_len, mask_prob=self.mask_prob, data_dir=self.data_dir,
                                         neg_sample_size=self.neg_sample_size)
        # Test 데이터셋 생성
        if stage == "test" or stage is None:
            self.movie_test = MovieLens(mode="Test", max_len=self.max_len, mask_prob=self.mask_prob, data_dir=self.data_dir,
                                        neg_sample_size=self.neg_sample_size)

    def train_dataloader(self):
        return DataLoader(self.movie_train, batch_size=self.batch_size, shuffle=True, pin_memory=self.pin_memory, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.movie_valid, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.movie_test, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory, num_workers=self.num_workers)
    # 인스턴스 없이도 확인할 수 있도록 static method로 설정
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument('--max_len', type=int, default=100)
        parser.add_argument('--mask_prob', type=float, default=0.2)
        parser.add_argument('--neg_sample_size', type=int, default=100)
        parser.add_argument('--data_dir', type=str, default=MOVIE_LENS)
        # DataLoader 관련
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--pin_memory', type=bool, default=True)
        parser.add_argument('--num_workers', type=int, default=4)
        # ml - 1m
        parser.add_argument('--item_size', type=int, default=3706)
        
        return parser
