import os,sys,argparse,json
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
from matcha.utils.utils import intersperse
from matcha.utils.utils import dict_to_attrdic



# def load_model(checkpoint_path):
#     model = MatchaTTS(**model_param).cuda()
#     model.load_state_dict(torch.load(checkpoint_path, map_location=device))
#     # model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device)
#     model.eval()
#     return model
# count_params = lambda x: f"{sum(p.numel() for p in x.parameters()):,}"


# model = load_model(MATCHA_CHECKPOINT)
# print(f"Model loaded! Parameter count: {count_params(model)}")

# def load_vocoder(checkpoint_path):
#     h = AttrDict(v1)
#     hifigan = HiFiGAN(h).to(device)
#     hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)['generator'])
#     _ = hifigan.eval()
#     hifigan.remove_weight_norm()
#     return hifigan

# vocoder = load_vocoder(HIFIGAN_CHECKPOINT)
# denoiser = Denoiser(vocoder, mode='zeros')


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
def synthesise(text, spks=None, n_timesteps=10, length_scale=1.0, temperature=0.667):
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
    
# def save_to_folder(filename: str, output: dict, folder: str):
#     folder = Path(folder)
#     folder.mkdir(exist_ok=True, parents=True)
#     np.save(folder / f'{filename}', output['mel'].cpu().numpy())
#     sf.write(folder / f'{filename}.wav', output['waveform'], 22050, 'PCM_24')



# texts = [
#     "The Secret Service believed that it was very doubtful that any President would ride regularly in a vehicle with a fixed top, even though transparent.",
#     "hi, how are you? can you help me? i want to go to school."
# ]

# ## Number of ODE Solver steps
# n_timesteps = 10

# ## Changes to the speaking rate
# length_scale=1.0

# ## Sampling temperature
# temperature = 0.667


# outputs, rtfs = [], []
# rtfs_w = []
# for i, text in enumerate(tqdm(texts)):
#     output = synthesise(text) #, torch.tensor([15], device=device, dtype=torch.long).unsqueeze(0))
#     output['waveform'] = to_waveform(output['mel'], vocoder)

#     # Compute Real Time Factor (RTF) with HiFi-GAN
#     t = (dt.datetime.now() - output['start_t']).total_seconds()
#     rtf_w = t * 22050 / (output['waveform'].shape[-1])

#     ## Pretty print
#     print(f"{'*' * 53}")
#     print(f"Input text - {i}")
#     print(f"{'-' * 53}")
#     print(output['x_orig'])
#     print(f"{'*' * 53}")
#     print(f"Phonetised text - {i}")
#     print(f"{'-' * 53}")
#     print(output['x_phones'])
#     print(f"{'*' * 53}")
#     print(f"RTF:\t\t{output['rtf']:.6f}")
#     print(f"RTF Waveform:\t{rtf_w:.6f}")
#     rtfs.append(output['rtf'])
#     rtfs_w.append(rtf_w)

#     ## Display the synthesised waveform
#     # ipd.display(ipd.Audio(output['waveform'], rate=22050))

#     ## Save the generated waveform
#     save_to_folder(i, output, OUTPUT_FOLDER)

# print(f"Number of ODE steps: {n_timesteps}")
# print(f"Mean RTF:\t\t\t\t{np.mean(rtfs):.6f} ± {np.std(rtfs):.6f}")
# print(f"Mean RTF Waveform (incl. vocoder):\t{np.mean(rtfs_w):.6f} ± {np.std(rtfs_w):.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MatchaTTS')
    parser.add_argument('--config', required=True, type=str, help='configs/xxx.json')
    parser.add_argument('--ckpt', required=True, type=str, help='checkpoint of matcha')
    parser.add_argument('--hifigan', type=str, default='checkpoints/hifigan_T2_v1')
    parser.add_argument('--spks', type=int, default=None, help='None for ljspeech, 0-109 for vctk')

    parser.add_argument('--text', type=str, default='The Secret Service believed that it was very doubtful that any President would ride regularly in a vehicle with a fixed top, even though transparent.')
    parser.add_argument('--output', type=str, default='output_tts.wav')
    args = parser.parse_args()
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.output = os.path.abspath(args.output)

    cfg = json.load(open(args.config))
    cfg = dict_to_attrdic(cfg)

    if cfg.model_param.encoder.encoder_params.n_spks > 1 and args.spks is None:
        print('ERROR: --spks can not be None when run multi-speaker mode')
        exit()
    if cfg.model_param.encoder.encoder_params.n_spks == 1 and not args.spks is None:
        print('ERROR: --spks must be None when run single-speaker mode')
        exit()
    if not args.spks is None:
        args.spks = torch.tensor([args.spks], dtype=torch.int64)
        args.spks = args.spks.to(device)

    model = MatchaTTS(**cfg.model_param).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    hifigan = HiFiGAN(dict_to_attrdic(v1)).to(device)
    hifigan.load_state_dict(torch.load(args.hifigan, map_location=device)['generator'])
    hifigan.eval()
    hifigan.remove_weight_norm()
    vocoder = hifigan
    denoiser = Denoiser(vocoder, mode='zeros')

    output = synthesise(args.text, spks=args.spks, n_timesteps=10, length_scale=1.0, temperature=0.667)
    output['waveform'] = to_waveform(output['mel'], vocoder)

    filedir_path = os.path.dirname(args.output)
    os.makedirs(filedir_path, exist_ok=True)
    sf.write(args.output, output['waveform'], 22050, 'PCM_24')
    print(f'output: {args.output}')