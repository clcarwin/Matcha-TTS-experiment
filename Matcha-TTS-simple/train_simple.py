import matcha
from matcha.data.text_mel_datamodule import TextMelDataModule
from matcha.models.matcha_tts import MatchaTTS
from matcha.utils.utils import dict_to_attrdic

import torch
import os,sys,time,random,datetime,json,argparse
import multiprocessing



def dt():
    return datetime.datetime.now().strftime('%H:%M:%S')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NS2VC v2')
    parser.add_argument('--config', required=True, type=str, help='configs/xxx.json')
    args = parser.parse_args()
    multiprocessing.set_start_method('spawn')

    cfg = json.load(open(args.config))
    cfg = dict_to_attrdic(cfg) # convert dict to AttrDict, data can be accessed as attribute
    # print(cfg)

    TextMelData = TextMelDataModule(**cfg.data_param)
    dataloader = TextMelData.train_dataloader()

    model = MatchaTTS(**cfg.model_param).cuda()
    # print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    config_name = os.path.basename(args.config).replace('.json','')
    model_savepath = f'logs/simple_{config_name}'
    os.makedirs(model_savepath, exist_ok=True)

    stepindex = 0
    loss_total,loss_count = 0,0
    while True:
        for data in dataloader:
            for k in data:
                if data[k] is not None: data[k] = data[k].cuda()

            dur_loss, prior_loss, diff_loss = model(**data)
            # print(dur_loss, prior_loss, diff_loss)

            loss = dur_loss + prior_loss + diff_loss

            loss_total += loss.item()
            loss_count += 1
            if 0==stepindex%1000:
                print(f'{dt()} step:{stepindex:08d} loss:{loss_total/loss_count:.4f}')
                loss_total,loss_count = 0,0
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if 0==stepindex%10000:
                torch.save(model.state_dict(),f'{model_savepath}/model_{config_name}_{stepindex:08d}.pt')
            
            stepindex += 1
        
