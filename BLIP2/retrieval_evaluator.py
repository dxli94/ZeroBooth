from dataset import coco_karpathy_retrieval_eval
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
import utils

def collate_fn(batch):
    image_list, text_list = [], []
    for image, text in batch:
        image_list.append(image)
        text_list += text
    return torch.stack(image_list,dim=0), text_list  


class retrieval_evaluator(object):
    def __init__(self, transform, image_root, batch_size=128, dataset='coco',split='test',k=128):
        self.batch_size = batch_size
        self.dataset = coco_karpathy_retrieval_eval(transform,image_root,ann_root='annotation',split=split,dataset=dataset)
        self.dataloader = DataLoader(self.dataset, 
                                     batch_size=batch_size,
                                     num_workers=5,
                                     collate_fn=collate_fn,
                                     shuffle=False,
                                     drop_last=False)  
        self.k = k
        
    @torch.no_grad()    
    def evaluate_itc(self, model, device):
        print("Evaluation on image-text retrieval")
        model.eval() 
        image_features = []
        text_features = []

        print("compute features ...")
        for image, text in tqdm(self.dataloader):
            image = image.to(device)
            image_feats, text_feats = model.compute_features(image=image, text=text)
            image_features.append(image_feats)
            text_features.append(text_feats)

        image_features = torch.cat(image_features,dim=0)
        text_features = torch.cat(text_features,dim=0)
        
        print("compute similarity ...")
        #sim = image_features @ text_features.t()
        sim = model.compute_similarity(image_features,text_features)
        
        print("compute recall ...")
        metric = compute_metric(sim.cpu().numpy(), sim.t().cpu().numpy(), self.dataset.txt2img, self.dataset.img2txt)
        return metric
    
    
    
    @torch.no_grad()    
    def evaluate_itm(self, model, device, blip='v2'):
        print("Evaluation on image-text retrieval")
        model.eval() 
        image_features = []
        image_embeds = []
        text_features = []
        texts = []
        
        print("compute features ...")
        for image, text in self.dataloader:
            image = image.to(device)
            image_feats, text_feats, embeds, tokens = model.compute_itc(image=image, text=text)
            image_features.append(image_feats)
            text_features.append(text_feats) 
            image_embeds.append(embeds.cpu())
            texts += text
            

        image_features = torch.cat(image_features,dim=0)
        text_features = torch.cat(text_features,dim=0)
        image_embeds = torch.cat(image_embeds,dim=0)

        print("compute itc similarity ...")
        
        if blip=='v1':
            sim = image_features @ text_features.t()
        else:
            sim = []
            for image_feat in image_features:
                sim_r2t = image_feat @ text_features.t()
                #sim_i2t = sim_r2t.mean(0)
                sim_i2t, _ = sim_r2t.max(0)
                sim.append(sim_i2t)
            sim = torch.stack(sim,dim=0)
        
        print("compute image-text similarity ...")
        
        num_tasks = utils.get_world_size()
        rank = utils.get_rank() 
        
        step = sim.size(0)//num_tasks + 1
        start = rank*step
        end = min(sim.size(0),start+step)
        
        score_i2t = torch.ones_like(sim)*-100

        for i,sims in enumerate(sim[start:end]):
            topk_sim, topk_idx = sims.topk(k=self.k, dim=0)
            image_embedding = image_embeds[start+i].repeat(self.k,1,1).to(device,non_blocking=True)           
            text = [texts[idx] for idx in topk_idx]
            score = model.compute_itm(image_embeds=image_embedding, text=text).float()
            score_i2t[start+i,topk_idx] = score + topk_sim  
        
        sim = sim.t()
        step = sim.size(0)//num_tasks + 1
        start = rank*step
        end = min(sim.size(0),start+step)
        
        score_t2i = torch.ones_like(sim)*-100
        
        print("compute text-image similarity ...")
        for i,sims in enumerate(sim[start:end]):
            topk_sim, topk_idx = sims.topk(k=self.k, dim=0)
            image_embedding = image_embeds[topk_idx].to(device,non_blocking=True)                         
            text = [texts[start+i]]*self.k
            score = model.compute_itm(image_embeds=image_embedding, text=text).float()
            score_t2i[start+i,topk_idx] = score + topk_sim  
            
        torch.distributed.barrier()   
        torch.distributed.all_reduce(score_i2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_t2i, op=torch.distributed.ReduceOp.SUM)  
        
        print("compute recall ...")
        metric = compute_metric(score_i2t.cpu().numpy(), score_t2i.cpu().numpy(), self.dataset.txt2img, self.dataset.img2txt)
        return metric
    
    

@torch.no_grad()
def compute_metric(score_i2t, score_t2i, txt2img, img2txt):
    
    #Images->Text 
    ranks = np.zeros(score_i2t.shape[0])
    for index,score in enumerate(score_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    #Text->Images 
    ranks = np.zeros(score_t2i.shape[0])
    
    for index,score in enumerate(score_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3

    metric =  {'txt_r1': tr1,
                'txt_r5': tr5,
                'txt_r10': tr10,
                'txt_r_mean': tr_mean,
                'img_r1': ir1,
                'img_r5': ir5,
                'img_r10': ir10,
                'img_r_mean': ir_mean
              }
    return metric    