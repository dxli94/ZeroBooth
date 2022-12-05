import os
import json
import random
import json
from PIL import Image
import re
from glob import glob
from copy import deepcopy

import torch
from torch.utils.data import Dataset, IterableDataset

import webdataset as wds

from torchvision.datasets.utils import download_url



class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, transform2=None): 
        
        self.annotation = []
        for f in ann_file:
            print('loading '+f)
            self.annotation += json.load(open(f,'r'))
        
        self.transform = transform 

    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index] 
      
        image = Image.open(ann['image']).convert('RGB')   
        image1 = self.transform(image)
        if type(ann['caption']) is list:    
            caption = random.choice(ann['caption'][:2])
        else:
            caption = ann['caption'].rstrip(' ').rstrip('.').lower()    
        return image1, caption    
    
    
    

class WDSDataset(IterableDataset):
    def __init__(self, location, preprocess):
        
        filelist = glob(os.path.join(location,'*','*.tar'))
        print("number of shards %d"%len(filelist))
        
        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(filelist),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode('pilrgb', handler=wds.warn_and_continue),
            wds.to_tuple('jpg', 'json', handler=wds.warn_and_continue),
            wds.map(preprocess),
        )

    def __iter__(self):
        return iter(self.inner_dataset)


    
class coco_karpathy_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, dataset='coco'):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        
        if dataset=='coco':
            urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                    'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
            filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        elif dataset=='flickr':
            urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_val.json',
                    'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json'}
            filenames = {'val':'flickr30k_val.json','test':'flickr30k_test.json'}
        
        download_url(urls[split],ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root      

        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):  
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root, ann['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  
        
        captions = [pre_caption(caption) for caption in ann['caption']]
        return image, captions
    
    
    
    
class coco_karpathy_caption_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        
        download_url(urls[split],ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]
        
        return image, int(img_id)   
    
class nocaps_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split):   
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/nocaps_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/nocaps_test.json'}
        filenames = {'val':'nocaps_val.json','test':'nocaps_test.json'}
        
        download_url(urls[split],ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):  
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        return image, int(ann['img_id'])   
    
    
        
    
class vqa_dataset(Dataset):
    def __init__(self, transform, vqa_ann, vqa_image_root, split='train'):

        self.transform = transform
        self.annotation = json.load(open(vqa_ann,'r'))    
        self.image_root = vqa_image_root  
        self.split = split

        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'])    
        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)    
        
        question = ann['question'].lower()
        
        if self.split == 'test':
            question_id = ann['question_id'] 
            if 'answer' in ann.keys():
                return image, question, question_id, ann['answer']
            else:    
                return image, question, question_id, ""


        elif self.split=='train':                       
           
            answer_weight = {}
            
            for answer in ann['answer']:
                if answer in answer_weight.keys():
                    answer_weight[answer] += 1/len(ann['answer'])
                else:
                    answer_weight[answer] = 1/len(ann['answer'])
                
            answers = list(answer_weight.keys())
            weights = list(answer_weight.values())  
            
            if len(answers)>5:
                data = list(zip(answers, weights))
                random.shuffle(data)
                answers, weights = zip(*data)
                answers = answers[:5]
                weights = weights[:5]

            return image, question, answers, weights
        
        
def pre_caption(caption):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')
            
    return caption

def pre_question(question):
    question = re.sub(
        r"([.!\"()*#:;~])",
        '',
        question.lower(),
    ) 
    question = question.rstrip(' ')
            
    return question

def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n          