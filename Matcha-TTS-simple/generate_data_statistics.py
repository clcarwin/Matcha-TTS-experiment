import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm
from matcha.data.text_mel_datamodule import TextMelDataModule


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


data_param = {
    "name": "ljspeech",
    "train_filelist_path": "assets/filelists/ljs_audio_text_train_filelist.txt",
    "valid_filelist_path": "assets/filelists/ljs_audio_text_val_filelist.txt",
    "batch_size": 32,
    "num_workers": 20,
    "pin_memory": True,
    "cleaners": ["english_cleaners2"],
    "add_blank": True,
    "n_spks": 1,
    "n_fft": 1024,
    "n_feats": 80,
    "sample_rate": 22050,
    "hop_length": 256,
    "win_length": 1024,
    "f_min": 0,
    "f_max": 8000,
    "data_statistics":  # Computed for ljspeech dataset
    {"mel_mean": -5.536622,
    "mel_std": 2.116101},
    "seed": 42
}

def main():
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
