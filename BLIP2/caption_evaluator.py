from dataset import coco_karpathy_caption_eval, nocaps_eval
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from data.utils import coco_caption_eval
import json
import os

class nocaps_evaluator(object):
    def __init__(self, transform, image_root,result_dir, batch_size=128, split='val'):
        self.batch_size = batch_size
        self.dataset = nocaps_eval(transform,image_root,ann_root='annotation',split=split)
        self.dataloader = DataLoader(self.dataset, 
                                     batch_size=batch_size,
                                     num_workers=5,
                                     shuffle=False,
                                     drop_last=False)  
        self.result_dir = result_dir
        self.split = split
        
    @torch.no_grad()    
    def evaluate(self, model, device):
        model.eval() 

        print("caption generation ...")
        
        result = []
        
        for image, image_id in tqdm(self.dataloader):
            image = image.to(device,non_blocking=True)
            captions = model.generate(image)
                
            for caption, img_id in zip(captions, image_id):
                result.append({"image_id": img_id.item(), "caption": caption})
                
        result_file = os.path.join(self.result_dir,'nocaps_%s.json'%self.split)        
        json.dump(result,open(result_file,'w'))
        return
                
class caption_evaluator(object):
    def __init__(self, transform, image_root, coco_gt_root, result_dir, batch_size=128, split='test'):
        self.batch_size = batch_size
        self.dataset = coco_karpathy_caption_eval(transform,image_root,ann_root='annotation',split=split)
        self.dataloader = DataLoader(self.dataset, 
                                     batch_size=batch_size,
                                     num_workers=5,
                                     shuffle=False,
                                     drop_last=False)  
        self.coco_gt_root = coco_gt_root
        self.result_dir = result_dir
        self.split = split
        
    @torch.no_grad()    
    def evaluate(self, model, device):
        model.eval() 

        print("caption generation ...")
        
        result = []
        
        for image, image_id in tqdm(self.dataloader):
            image = image.to(device,non_blocking=True)
            captions = model.generate(image)
                
            for caption, img_id in zip(captions, image_id):
                result.append({"image_id": img_id.item(), "caption": caption})
        
        result_file = os.path.join(self.result_dir,'%s.json'%self.split)
        json.dump(result,open(result_file,'w'))
        metric = coco_caption_eval(self.coco_gt_root,result_file,self.split)
        
        return metric
    
    
    
