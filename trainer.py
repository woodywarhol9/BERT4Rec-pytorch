import argparse
import pytorch_lightning as pl
import datetime

from datamodule import DataModule
from lit_model import BERT4REC
# 한 모듈에서만 사용될 경우 _로 설정. private 같은 역할.
def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(description="BERT4REC")
    # Trainer args 추가
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    # pl.Trainer를 상속하면서 Trainer api에 있는 것들 argparse 사용 가능.
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])
    # Get data arguments
    data_group = parser.add_argument_group("Data Args")
    DataModule.add_to_argparse(data_group)
    # Get model arguments
    model_group = parser.add_argument_group("Model Args")
    BERT4REC.add_to_argparse(model_group)
    return parser

def _set_trainer_args(args):
    """
    Trainer args를 지정.
    """
    args.weights_summary = "full"
    args.gpus = 1
    args.max_epochs = 100
    args.track_grad_norm = 2
    args.detect_anomaly = True
    args.gradient_clip_val = 5.0
    args.profiler = "simple"
    args.gradient_clip_algorithm = "norm"
    args.save_dir = "Training/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def main():
    """
    Run an experiment.
    Sample command:
    ```
    python training/run_experiment.py --max_epochs=3 --gpus='0,' --num_workers=20 --model_class=MLP --data_class=MNIST
    ```
    """
    parser = _setup_parser()
    args = parser.parse_args()
    _set_trainer_args(args)

    data = DataModule(args)
    lit_model = BERT4REC(args)

    logger = pl.loggers.TensorBoardLogger(save_dir = args.save_dir, name = "BERT4REC")
    early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=100)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=[lr_monitor, early_stopping], weights_save_path=args.save_dir)

    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)
    # pylint: enable=no-member
if __name__ == "__main__":
    main()