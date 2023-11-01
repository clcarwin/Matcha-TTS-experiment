import datetime as dt
from pathlib import Path

# import IPython.display as ipd
import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm

# Hifigan imports
from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN
# Matcha imports
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.model import denormalize
from matcha.utils.utils import get_user_data_dir, intersperse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MATCHA_CHECKPOINT = "logs/simple/model_00020000.pt"
HIFIGAN_CHECKPOINT = "checkpoints/hifigan_T2_v1"
OUTPUT_FOLDER = "logs/simple"




data_param = {
    "name": "ljspeech",
    "train_filelist_path": "data/filelists/ljs_audio_text_train_filelist.txt",
    "valid_filelist_path": "data/filelists/ljs_audio_text_val_filelist.txt",
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














def load_model(checkpoint_path):
    model = MatchaTTS(**model_param).cuda()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    return model
count_params = lambda x: f"{sum(p.numel() for p in x.parameters()):,}"


model = load_model(MATCHA_CHECKPOINT)
print(f"Model loaded! Parameter count: {count_params(model)}")

def load_vocoder(checkpoint_path):
    h = AttrDict(v1)
    hifigan = HiFiGAN(h).to(device)
    hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)['generator'])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan

vocoder = load_vocoder(HIFIGAN_CHECKPOINT)
denoiser = Denoiser(vocoder, mode='zeros')


@torch.inference_mode()
def process_text(text: str):
    x = torch.tensor(intersperse(text_to_sequence(text, ['english_cleaners2']), 0),dtype=torch.long, device=device)[None]
    x_lengths = torch.tensor([x.shape[-1]],dtype=torch.long, device=device)
    x_phones = sequence_to_text(x.squeeze(0).tolist())
    return {
        'x_orig': text,
        'x': x,
        'x_lengths': x_lengths,
        'x_phones': x_phones
    }


@torch.inference_mode()
def synthesise(text, spks=None):
    text_processed = process_text(text)
    start_t = dt.datetime.now()
    output = model.synthesise(
        text_processed['x'], 
        text_processed['x_lengths'],
        n_timesteps=n_timesteps,
        temperature=temperature,
        spks=spks,
        length_scale=length_scale
    )
    # merge everything to one dict    
    output.update({'start_t': start_t, **text_processed})
    return output

@torch.inference_mode()
def to_waveform(mel, vocoder):
    audio = vocoder(mel).clamp(-1, 1)
    audio = denoiser(audio.squeeze(0), strength=0.00025).cpu().squeeze()
    return audio.cpu().squeeze()
    
def save_to_folder(filename: str, output: dict, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    np.save(folder / f'{filename}', output['mel'].cpu().numpy())
    sf.write(folder / f'{filename}.wav', output['waveform'], 22050, 'PCM_24')



texts = [
    "The Secret Service believed that it was very doubtful that any President would ride regularly in a vehicle with a fixed top, even though transparent.",
    "hi, how are you? can you help me? i want to go to school."
]

## Number of ODE Solver steps
n_timesteps = 10

## Changes to the speaking rate
length_scale=1.0

## Sampling temperature
temperature = 0.667


outputs, rtfs = [], []
rtfs_w = []
for i, text in enumerate(tqdm(texts)):
    output = synthesise(text) #, torch.tensor([15], device=device, dtype=torch.long).unsqueeze(0))
    output['waveform'] = to_waveform(output['mel'], vocoder)

    # Compute Real Time Factor (RTF) with HiFi-GAN
    t = (dt.datetime.now() - output['start_t']).total_seconds()
    rtf_w = t * 22050 / (output['waveform'].shape[-1])

    ## Pretty print
    print(f"{'*' * 53}")
    print(f"Input text - {i}")
    print(f"{'-' * 53}")
    print(output['x_orig'])
    print(f"{'*' * 53}")
    print(f"Phonetised text - {i}")
    print(f"{'-' * 53}")
    print(output['x_phones'])
    print(f"{'*' * 53}")
    print(f"RTF:\t\t{output['rtf']:.6f}")
    print(f"RTF Waveform:\t{rtf_w:.6f}")
    rtfs.append(output['rtf'])
    rtfs_w.append(rtf_w)

    ## Display the synthesised waveform
    # ipd.display(ipd.Audio(output['waveform'], rate=22050))

    ## Save the generated waveform
    save_to_folder(i, output, OUTPUT_FOLDER)

print(f"Number of ODE steps: {n_timesteps}")
print(f"Mean RTF:\t\t\t\t{np.mean(rtfs):.6f} ± {np.std(rtfs):.6f}")
print(f"Mean RTF Waveform (incl. vocoder):\t{np.mean(rtfs_w):.6f} ± {np.std(rtfs_w):.6f}")