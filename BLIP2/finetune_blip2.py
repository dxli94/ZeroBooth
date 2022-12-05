'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from models.blipv2_ft import blip
import utils
from utils import cosine_lr_schedule, warmup_lr_schedule
from data import create_dataset, create_sampler, create_loader
from constant import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

from retrieval_evaluator import retrieval_evaluator
from caption_evaluator import caption_evaluator, nocaps_evaluator

def train(model, data_loader, optimizer, scaler, epoch, device):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image, caption, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device)     
        
        if epoch==0:
            warmup_lr_schedule(optimizer, i, config['warmup_steps'], config['warmup_lr'], config['init_lr'])  
            
        with torch.cuda.amp.autocast(enabled=scaler is not None):             
            loss_itc,loss_itm,loss_lm = model(image, caption, idx)  
            loss = 0.5*loss_itc + loss_itm + loss_lm
            
            loss = loss/config['acc_grad_iters']
        
        if scaler is not None:
            scaler.scale(loss).backward()  
        else:
            loss.backward()        
            
        if (i+1)%config['acc_grad_iters']==0:         
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update() 
            else:
                optimizer.step()     
            optimizer.zero_grad()
            
        metric_logger.update(loss_itc=loss_itc.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_lm=loss_lm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  



def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")
    normalize = transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)    
    train_dataset = create_dataset('retrieval_%s'%config['dataset'], config, normalize, min_scale=0.5)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank)         
    else:
        samplers = [None]
    
    train_loader = create_loader([train_dataset],samplers,batch_size=[config['batch_size']],num_workers=[4],
                                                          is_trains=[True], collate_fns=[None])[0]         
    transform_test = transforms.Compose([
        transforms.Resize(config['image_size'],interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(config['image_size']),
        transforms.ToTensor(),
        normalize,
        ])  
    
    eval_val = retrieval_evaluator(transform_test, config['image_root'], dataset=config['dataset'],split='val',k=config['k'])
    eval_test = retrieval_evaluator(transform_test, config['image_root'], dataset=config['dataset'],split='test',k=config['k'])
    if utils.is_main_process():                         
        c_eval = caption_evaluator(transform_test, config['image_root'], config['coco_gt_root'], args.result_dir, batch_size=64)
        #c_eval = nocaps_evaluator(transform_test, config['nocaps_image_root'],args.result_dir, batch_size=64, split='val')
        
    #### Model #### 
    print("Creating model")
    model = blip(config=config)
    model = model.to(device)   
    
    #### Optimizer #### 
    num_parameters = 0
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)
        num_parameters += p.data.nelement()     

    optim_params = [{"params": p_wd, "weight_decay": config['weight_decay']},
                    {"params": p_non_wd, "weight_decay": 0}]    
    optimizer = torch.optim.AdamW(optim_params, lr=config['init_lr'], weight_decay=config['weight_decay'], betas=(0.9,0.98)) 
    print("number of trainable parameters: %d"%num_parameters)
    
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']    
        msg = model.load_state_dict(state_dict,strict=False)                            
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
        
    best = 0
    best_epoch = 0
    
    if args.evaluate:    
#         test_result = eval_test.evaluate_itm(model_without_ddp, device)
#         print(test_result)         
        if utils.is_main_process():             
            caption_result = c_eval.evaluate(model_without_ddp, device)
            print(caption_result) 
            
            
    print("Start training")
    start_time = time.time()    
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                
            cosine_lr_schedule(optimizer, epoch, config['max_epoch']-1, config['init_lr'], config['min_lr'])
                
            train_stats = train(model, train_loader, optimizer, scaler, epoch, device) 
 
        val_result = eval_val.evaluate_itm(model_without_ddp, device)
        print(val_result) 
        test_result = eval_test.evaluate_itm(model_without_ddp, device)
        print(test_result) 
        
        if utils.is_main_process():   
            caption_result = c_eval.evaluate(model_without_ddp, device)
            print(caption_result)
            if args.evaluate:            
                log_stats = {**{f'{k}': v for k, v in val_result.items()},    
                             **{f'{k}': v for k, v in test_result.items()},                  
                            }
                with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")                   
            else:             
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch)) 
                if val_result['txt_r_mean'] + val_result['img_r_mean'] > best:
                    best = val_result['txt_r_mean'] + val_result['img_r_mean']
                    best_epoch = epoch                
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth')) 
                    
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_result.items()},    
                             **{f'text_{k}': v for k, v in test_result.items()},  
                             **{f'{k}': v for k, v in caption_result.eval.items()},    
                             'epoch': epoch,
                             'best_epoch': best_epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")     
                    
        if args.evaluate: 
            break
        dist.barrier()     

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/finetune.yaml')
    parser.add_argument('--output_dir', default='output_blip2/finetune/clip_im392_lr5e6_lm_0.5itc')    
    parser.add_argument('--checkpoint', default='')    
    parser.add_argument('--amp', action='store_true')    
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)