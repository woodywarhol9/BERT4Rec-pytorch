import argparse
import datetime

from datamodule import DataModule
from lit_model import BERT4REC

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor


def _setup_parser():
    """
    Model, Trainer, Data arguments 정의
    """
    parser = argparse.ArgumentParser(description="BERT4REC")
    # Trainer args
    trainer_parser = Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"
    # pl.Trainer_parser inheritance
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])
    # Get data arguments
    data_group = parser.add_argument_group("Data Args")
    DataModule.add_to_argparse(data_group)
    # Get model arguments
    model_group = parser.add_argument_group("Model Args")
    BERT4REC.add_to_argparse(model_group)
    
    return parser


def _set_trainer_args(args) -> None:
    """
    Trainer args 설정
    """
    # 학습 관련 설정
    args.gpus = 1
    args.max_epochs = 100
    args.gradient_clip_val = 5.0
    args.gradient_clip_algorithm = "norm"
    # logger 설정
    args.save_dir = "Training/logs/" + \
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 현재 시각 저장
    args.name = "BERT4REC"
    # EarlyStopping 설정
    args.monitor = "val_loss"
    args.mode = "min"
    args.patience = 100
    # LearningRateMonitor 설정
    args.logging_interval = "step"
    # check point dir
    args.weights_save_path = args.save_dir
    # 디버깅 설정
    args.weights_summary = "full"
    args.track_grad_norm = 2
    args.detect_anomaly = True
    args.profiler = "simple"


def main():
    """
    arguments 받아와서 fit(train + valid), test 진행
    """
    # arguments 설정
    parser = _setup_parser()
    args = parser.parse_args()
    _set_trainer_args(args)
    # DataModule, Model 불러오기
    data = DataModule(args)
    lit_model = BERT4REC(args)
    # logger 및 callbacks 설정
    logger = TensorBoardLogger(save_dir=args.save_dir, name=args.name)
    early_stopping = EarlyStopping(monitor=args.monitor, mode=args.mode, patience=args.patience)
    lr_monitor = LearningRateMonitor(logging_interval=args.logging_interval)
    # Trainer 정의 및 fit(train + valid)/test 실행
    trainer = Trainer.from_argparse_args(args, logger=logger, callbacks=[early_stopping, lr_monitor], weights_save_path=args.weights_save_path)
    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)


if __name__ == "__main__":
    main()
