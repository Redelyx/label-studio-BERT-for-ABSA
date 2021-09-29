# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team and authors from University of Illinois at Chicago.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#define your pre-trained (post-trained) models here with their paths.
import torch
import os


#at_rest
ae_rest_dropout = 0.0
ae_rest_epsilon = 5.0
asc_rest_dropout = 0.5
asc_rest_epsilon = 5.0
#at_laptop
ae_lap_dropout = 0.0
ae_lap_epsilon = 0.2
asc_lap_dropout = 0.4
asc_lap_epsilon = 5.0


MODEL_ARCHIVE_MAP = {
    'laptop_pt': 'pt_model/laptop_pt/',  
    'rest_pt': 'pt_model/rest_pt/',      
    'bert-base': 'pt_model/bert-base/'   
}


def find_model():

    files = [] 

    for file in os.listdir('.'):
        if file.endswith(".pt"):
            files.append(file[:-3])
    items_ae = []
    items_asc = []
    for file in files:
        items = file.split('_')  #model_task_domain
        print(items)
        if items[1] == 'ae' and items_ae == []:
            items_ae = items
        if items[1] == 'asc' and items_asc == []:
            items_asc = items
        
    return items_ae, items_asc


def pick_model(bert_task, bert_model, label_list):
    items_ae, items_asc = find_model()
    if(bert_task == "ae"):
        btype = items_ae[0]
    else:
        btype = items_asc[0]

    if btype == "at":
        if bert_task == "ae":
            from berts.bat_ae import BertForABSA
            model = BertForABSA.from_pretrained(bert_model, num_labels = len(label_list), dropout=ae_lap_dropout, epsilon=ae_lap_epsilon) 
        elif bert_task == "asc":  
            from berts.bat_asc import BertForABSA  
            model = BertForABSA.from_pretrained(bert_model, num_labels = len(label_list), dropout=asc_lap_dropout, epsilon=asc_lap_epsilon)   
    elif btype == "psum":
        if bert_task == "ae":
            from berts.bpsum_ae import BertForABSA
        elif bert_task == "asc":  
            from berts.bpsum_asc import BertForABSA  
        model = BertForABSA.from_pretrained(bert_model, num_labels = len(label_list))
    elif btype == "hsum":
        if bert_task == "ae":
            from berts.bhsum_ae import BertForABSA
        elif bert_task == "asc":  
            from berts.bhsum_asc import BertForABSA 
        model = BertForABSA.from_pretrained(bert_model, num_labels = len(label_list))
    else:
        print("Task or model not valid.")
        return
    model.load_state_dict(torch.load(btype + "_" + bert_task + "_" + pick_domain() + ".pt"))
    return model
    
def pick_domain():
    items_ae, items_asc = find_model()
    bert = items_ae[2]
    return bert

