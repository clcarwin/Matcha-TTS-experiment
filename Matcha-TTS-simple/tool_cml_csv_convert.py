import os,argparse,sys,datetime,json


parser = argparse.ArgumentParser(description='CML-TTS')
parser.add_argument('--input', required=True, type=str, help='xx/xx/dev.csv test.csv train.csv')
# parser.add_argument('--val', required=True, type=str, help='dir xx/xx/val')
# parser.add_argument('--ckpt', default=None, type=str, help='load pretrain weight when not none')
# parser.add_argument('--finetune', action='store_true')
args = parser.parse_args()
print(args)

args.input = os.path.abspath(args.input)
currentdir = os.path.dirname(args.input)

csvlines = None
with open(args.input,'r') as fp:
    csvlines = fp.readlines()
csvlines.sort()

maxlen = 0

# get all id
orig_id_dict = {}
for k in csvlines:
    vlist = k.split('|')
    if len(vlist)!=8:
        continue

    wav_filename = vlist[0]
    transcript_wav2vec = vlist[3]

    if '.wav' not in wav_filename:
        continue

    orig_id = vlist[7].replace('\n','')
    orig_id_dict[orig_id] = 0
# map orig_id to new_id
idindex = 0
for k in orig_id_dict:
    orig_id_dict[k] = idindex
    idindex += 1

# print(orig_id_dict)
print(f'new_id range is 0-{idindex-1}')

# wav_filename|wav_filesize|transcript|transcript_wav2vec|levenshtein|duration|num_words|client_id
outputlines = []
for k in csvlines:
    vlist = k.split('|')
    if len(vlist)!=8:
        continue

    wav_filename = vlist[0]
    transcript_wav2vec = vlist[3]

    if '.wav' not in wav_filename:
        continue
    if float(vlist[5])<3:
        continue # duration must more than 3s
    
    if len(transcript_wav2vec)>maxlen:
        maxlen = len(transcript_wav2vec)
    
    orig_id = vlist[7].replace('\n','')

    line = f"{currentdir}/{wav_filename}|{orig_id_dict[orig_id]}|{transcript_wav2vec}"
    outputlines.append(line + '\n')

outputfile_path = f"{args.input}.processed.txt"
outputlines.sort()
with open(outputfile_path,'w') as fp:
    fp.writelines(outputlines)
