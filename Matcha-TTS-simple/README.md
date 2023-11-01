# Matcha-TTS-simple

## build
```bash
cd matcha/utils/monotonic_align
python setup.py build_ext --inplace
cd ../../..
```

## data
1. download LJSpeech-1.1.tar.bz2
2. download filelists from https://github.com/NVIDIA/tacotron2/tree/master/filelists
3. change wav path in ljs_audio_text_train_filelist.txt ljs_audio_text_val_filelist.txt

## train
```bash
python train_simple.py
# 22:34:53 step:00000000 loss:5.9429
# 22:38:27 step:00001000 loss:2.6250
# 23:16:01 step:00010000 loss:2.1116
# 23:57:32 step:00020000 loss:2.1995
# 01:17:34 step:00040000 loss:2.1542
# 03:58:40 step:00080000 loss:2.0707
# 08:41:37 step:00150000 loss:2.0971
```