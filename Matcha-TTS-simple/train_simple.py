import matcha
from matcha.data.text_mel_datamodule import TextMelDataModule
from matcha.models.matcha_tts import MatchaTTS

import torch
import os,sys,time,random,datetime
import multiprocessing

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

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

encoder_param = AttrDict({
    "encoder_type": "RoPE Encoder",
    "encoder_params":AttrDict({
        "n_feats": 80,
        "n_channels": 192,
        "filter_channels": 768,
        "filter_channels_dp": 256,
        "n_heads": 2,
        "n_layers": 6,
        "kernel_size": 3,
        "p_dropout": 0.1,
        "spk_emb_dim": 64,
        "n_spks": 1,
        "prenet": True
    }),
    "duration_predictor_params": AttrDict({
        "filter_channels_dp": 256,
        "kernel_size": 3,
        "p_dropout": 0.1,
    })
})


decoder_param = {
    "channels": [256, 256],
    "dropout": 0.05,
    "attention_head_dim": 64,
    "n_blocks": 1,
    "num_mid_blocks": 2,
    "num_heads": 2,
    "act_fn": "snakebeta"
}

cfm_param = AttrDict({
    "name": "CFM",
    "solver": "euler",
    "sigma_min": 1e-4
})

model_param = {
    "n_vocab":178,
    "n_spks":1,
    "spk_emb_dim":64,
    "n_feats":80,
    "encoder":encoder_param,
    "decoder":decoder_param,
    "cfm":cfm_param,
    "data_statistics":{"mel_mean": -5.536622,"mel_std": 2.116101},
    "out_size":None
}


def dt():
    return datetime.datetime.now().strftime('%H:%M:%S')

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    print(TextMelDataModule)
    TextMelData = TextMelDataModule(**data_param)
    TextMelData.setup()

    dataloader = TextMelData.train_dataloader()
    # print(dataloader)
    # for data in dataloader:
    #     print(data)

    model = MatchaTTS(**model_param).cuda()
    # print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    os.makedirs('logs/simple', exist_ok=True)
    stepindex = 0
    while True:
        for data in dataloader:
            # print('----')
            # for k in data: print(k)
            for k in data:
                if k!='spks': data[k] = data[k].cuda()

            dur_loss, prior_loss, diff_loss = model(**data)
            # print(dur_loss, prior_loss, diff_loss)

            loss = dur_loss + prior_loss + diff_loss
            if 0==stepindex%100:
                print(f'{dt()} step:{stepindex:08d} loss:{loss.item():.4f}')
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if 0==stepindex%10000:
                torch.save(model.state_dict(),f'logs/simple/model_{stepindex:08d}.pt')
            
            stepindex += 1
        
