import os,sys,inspect
from os.path import dirname
import modelconfig
import run_config as c   
import ipdb  

def pick_model(bert_task, bert_model, label_list):
    if c.btype == "at":
        if bert_task == "ae":
            from berts.bat_ae import BertForABSA
        elif bert_task == "asc":  
            from berts.bat_asc import BertForABSA  
        return BertForABSA.from_pretrained(bert_model, num_labels = len(label_list), dropout=c.ae_lap_dropout, epsilon=c.ae_lap_epsilon)   
    elif c.btype == "psum":
        if bert_task == "ae":
            from berts.bpsum_ae import BertForABSA
        elif bert_task == "asc":  
            from berts.bpsum_asc import BertForABSA  
        return BertForABSA.from_pretrained(bert_model, num_labels = len(label_list))
    elif c.btype == "hsum":
        if bert_task == "ae":
            from berts.bhsum_ae import BertForABSA
        elif bert_task == "asc":  
            from berts.bhsum_asc import BertForABSA 
        return BertForABSA.from_pretrained(bert_model, num_labels = len(label_list))
    else:
        print("Task or model not valid.")

def pick_domain():
    return c.bert
