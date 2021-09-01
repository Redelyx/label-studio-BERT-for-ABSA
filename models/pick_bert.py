import run_config as c   
import torch

def pick_model(bert_task, bert_model, label_list):
    if c.btype == "at":
        if bert_task == "ae":
            from berts.bat_ae import BertForABSA
        elif bert_task == "asc":  
            from berts.bat_asc import BertForABSA  
        model = BertForABSA.from_pretrained(bert_model, num_labels = len(label_list), dropout=c.ae_lap_dropout, epsilon=c.ae_lap_epsilon)   
    elif c.btype == "psum":
        if bert_task == "ae":
            from berts.bpsum_ae import BertForABSA
        elif bert_task == "asc":  
            from berts.bpsum_asc import BertForABSA  
        model = BertForABSA.from_pretrained(bert_model, num_labels = len(label_list))
    elif c.btype == "hsum":
        if bert_task == "ae":
            from berts.bhsum_ae import BertForABSA
        elif bert_task == "asc":  
            from berts.bhsum_asc import BertForABSA 
        model = BertForABSA.from_pretrained(bert_model, num_labels = len(label_list))
    else:
        print("Task or model not valid.")
        return
    model.load_state_dict(torch.load(c.btype+"_" + bert_task+"_" + c.domain + ".pt"))
    return model
def pick_domain():
    return c.bert
