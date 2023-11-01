import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm
from matcha.data.text_mel_datamodule import TextMelDataModule
from matcha.utils.utils import dict_to_attrdic


def compute_data_statistics(data_loader: torch.utils.data.DataLoader, out_channels: int):
    """Generate data mean and standard deviation helpful in data normalisation

    Args:
        data_loader (torch.utils.data.Dataloader): _description_
        out_channels (int): mel spectrogram channels
    """
    total_mel_sum = 0
    total_mel_sq_sum = 0
    total_mel_len = 0

    for batch in tqdm(data_loader, leave=False):
        mels = batch["y"]
        mel_lengths = batch["y_lengths"]

        total_mel_len += torch.sum(mel_lengths)
        total_mel_sum += torch.sum(mels)
        total_mel_sq_sum += torch.sum(torch.pow(mels, 2))

    data_mean = total_mel_sum / (total_mel_len * out_channels)
    data_std = torch.sqrt((total_mel_sq_sum / (total_mel_len * out_channels)) - torch.pow(data_mean, 2))

    return {"mel_mean": data_mean.item(), "mel_std": data_std.item()}




def main():
    parser = argparse.ArgumentParser(description='NS2VC v2')
    parser.add_argument('--config', required=True, type=str, help='configs/xxx.json')
    args = parser.parse_args()

    cfg = json.load(open(args.config))
    cfg = dict_to_attrdic(cfg)

    data_param = cfg.data_param
    data_param['batch_size'] = 256
    data_param['data_statistics'] = None
    data_param["seed"] = 1234

    print('Start')
    text_mel_datamodule = TextMelDataModule(**data_param)
    data_loader = text_mel_datamodule.train_dataloader()
    print("Dataloader loaded! Now computing stats...")
    params = compute_data_statistics(data_loader, data_param["n_feats"])
    print('')
    print(params)



if __name__ == "__main__":
    main()
