import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from datamodule import DataModule
from lit_model import BERT4REC

# 한 모듈에서만 사용될 경우 _로 설정. private 같은 역할.
def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(description = "BERT4REC")
    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    # pl.Trainer를 상속하면서 Trainer api에 있는 것들 argparse 사용 가능.
    parser = argparse.ArgumentParser(add_help = False, parents=[trainer_parser])
    # Get data arguments
    data_group = parser.add_argument_group("Data Args")
    DataModule.add_to_argparse(data_group)
    # Get model arguments
    model_group = parser.add_argument_group("Model Args")
    BERT4REC.add_to_argparse(model_group)
    # 나중에 추가  
    return parser

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
    data = DataModule(args)
    lit_model = BERT4REC(args)

    logger = pl.loggers.TensorBoardLogger("training/logs", name = "BERT4REC")
    #early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)
    #model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #    filename="{epoch:03d}-{val_loss:.3f}-{val_cer:.3f}", monitor="val_loss", mode="min"
    #
    # trainer flag 설정
    args.weights_summary = "full"  # Print full summary of the model
    args.gpus = 1
    args.max_epochs = 100
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, weights_save_path="training/logs")
    # batch_size, lr 검색
    #trainer.tune(lit_model, datamodule = data)  # If passing --auto_lr_find, this will set learning rate
    trainer.fit(lit_model, datamodule = data)
    trainer.test(lit_model, datamodule = data)
    # pylint: enable=no-member

if __name__ == "__main__":
    main()