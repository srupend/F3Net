import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
osenvs = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
import sys
import time
import torch
import torch.nn

from utils import evaluate, get_dataset, FFDataset, setup_logger
from trainer import Trainer
import numpy as np
import random

#for percentage bar
from tqdm import tqdm

# config
dataset_path = 'D:/Visual Studio/Srupen_2/Modification_2/dataset'
pretrained_path = 'pretrained/xception-b5690688.pth'
batch_size = 4
gpu_ids = [*range(osenvs)]
max_epoch = 1
loss_freq = 40
mode = 'FAD' # ['Original', 'FAD', 'LFS', 'Both', 'Mix']
ckpt_dir = '/data/yike/checkpoints/F3Net'
ckpt_name = 'FAD4_bz128'


if __name__ == '__main__':
    dataset = FFDataset(dataset_root=os.path.join(dataset_path, 'train', 'real'), size=299, frame_num=300, augment=True)
    dataloader_real = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size // 2,
        shuffle=True,
        num_workers=8)
    
    len_dataloader = dataloader_real.__len__()

    dataset_img, total_len =  get_dataset(name='train', size=299, root=dataset_path, frame_num=300, augment=True)
    dataloader_fake = torch.utils.data.DataLoader(
        dataset=dataset_img,
        batch_size=batch_size // 2,
        shuffle=True,
        num_workers=8
    )

    # init checkpoint and logger
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    logger = setup_logger(ckpt_path, 'result.log', 'logger')
    best_val = 0.
    ckpt_model_name = 'best.pkl'
    
    # train
    model = Trainer(gpu_ids, mode, pretrained_path)
    model.total_steps = 0
    epoch = 0
    
    while epoch < max_epoch:

        fake_iter = iter(dataloader_fake)
        real_iter = iter(dataloader_real)
        
        logger.debug(f'Epoch No::: {epoch}')
        i = 0
        progress_bar = tqdm(total=len_dataloader, desc=f"Epoch {epoch + 1}/{max_epoch}")
        while i < len_dataloader:
            
            i += 1
            model.total_steps += 1

            try:
                data_real = next(real_iter)
               # data_real = real_iter.next()
                # data_fake = fake_iter.next()
                data_fake = next(fake_iter)
            except StopIteration:
                break
            # -------------------------------------------------
            
            if data_real.shape[0] != data_fake.shape[0]:
                continue

            bz = data_real.shape[0]
            
            data = torch.cat([data_real,data_fake],dim=0)
            label = torch.cat([torch.zeros(bz).unsqueeze(dim=0),torch.ones(bz).unsqueeze(dim=0)],dim=1).squeeze(dim=0)

            # manually shuffle
            idx = list(range(data.shape[0]))
            random.shuffle(idx)
            data = data[idx]
            label = label[idx]

            data = data.detach()
            label = label.detach()

            model.set_input(data,label)
            loss = model.optimize_weight()

            if model.total_steps % loss_freq == 0:
                logger.debug(f'loss: {loss} at step: {model.total_steps}')

            if i % int(len_dataloader / 10) == 0:
                model.model.eval()
                auc, r_acc, f_acc = evaluate(model, dataset_path, mode='valid')
                logger.debug(f'(Val @ epoch {epoch}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}')
                auc, r_acc, f_acc = evaluate(model, dataset_path, mode='test')
                logger.debug(f'(Test @ epoch {epoch}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}')
                model.model.train()

            progress_bar.set_postfix({"Step": i+1})
            progress_bar.update(batch_size//2)

        epoch = epoch + 1
        progress_bar.close()

    model.model.eval()
    auc, r_acc, f_acc = evaluate(model, dataset_path, mode='test')
    logger.debug(f'(Test @ epoch {epoch}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}')
