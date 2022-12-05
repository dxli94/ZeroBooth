from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from dataset import vqa_dataset
from vqaTools.vqa import VQA
from vqaTools.vqaEval import VQAEval
from data.utils import save_result, vqa_eval

import utils

import json
import os
import spacy

class vqa_evaluator(object):
    def __init__(self, transform, image_root, ann, result_dir, vqa_gt=None, vqa_question=None, batch_size=128, split='test', name='vqa'):
        dataset = vqa_dataset(transform, ann, image_root, split)
       
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=utils.get_world_size(), rank=utils.get_rank()       , shuffle=False)
        self.data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=False,
            sampler=sampler,
            shuffle=False
        ) 
    
        self.result_dir = result_dir
        if name!='gqa':
            self.vqa = VQA(vqa_gt, vqa_question)        
            self.vqa_question = vqa_question        
        self.name = name
        
    @torch.no_grad()
    def evaluate(self, model, device, qa_prompt, past_key_value=None) :
        # test
        model.eval()
        
        nlp = spacy.load('en_core_web_sm')

        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Generate VQA result:'
        print_freq = 50

        result = []

        for n, (image,question,question_id,gt_answer) in enumerate(metric_logger.log_every(self.data_loader, print_freq, header)):        
            image = image.to(device,non_blocking=True)                

            prompt = [qa_prompt.format(q) for q in question]
            
            question_blip = [q.strip('?') for q in question]
            
            answers = model.generate_answer(image, prompt, question=question_blip, past_key_value=past_key_value)
 
      
            for i, (answer, ques_id, ques) in enumerate(zip(answers, question_id, question)):
                ques_id = int(ques_id.item())   
                
                if self.name=='okvqa':
                    doc = nlp(answer)

                    words = []
                    for token in doc:
                        if token.pos_ in ['NOUN','VERB']:
                            words.append(token.lemma_)
                        else:
                            words.append(token.text)
                    answer = " ".join(words)
                
                if self.name=='gqa':
                    result.append({"question_id":ques_id, "question":ques, "answer":answer, "gt_answer": gt_answer[i]})
                else:
                    result.append({"question_id":ques_id, "question":ques, "answer":answer})
                    
        result_file = save_result(result, self.result_dir, '%s_result'%self.name)  
        
        if utils.is_main_process():   
            if self.name=='gqa':
                vqaEval = VQAEval()
                acc = []
                result = json.load(open(result_file,'r'))
                for res in result:
                    gt_ans = res["gt_answer"]
                    pred = res["answer"]

                    pred = vqaEval.processPunctuation(pred)
                    pred = vqaEval.processDigitArticle(pred)

                    vqa_acc = 1 if pred == gt_ans else 0

                    acc.append(vqa_acc)
                accuracy = sum(acc) / len(acc) * 100
                print("GQA accuracy: %.2f"%accuracy)
            else:
                vqaEval = vqa_eval(self.vqa, result_file, self.vqa_question)
                accuracy = vqaEval.accuracy['overall']
        else:
            accuracy = None
            
        return accuracy
    
    
    
    
