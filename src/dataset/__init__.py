from .era_numpy import ECDataModule


def get_dataloader(configs):
    datamodule = ECDataModule(configs, configs.mode)

    return datamodule
