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
import run_config as c   
import torch

MODEL_ARCHIVE_MAP = {
    'laptop_pt': 'pt_model/laptop_pt/',  
    'rest_pt': 'pt_model/rest_pt/',      
    'bert-base': 'pt_model/bert-base/'   
}

def pick_model(bert_task, bert_model, label_list):
    if c.btype == "at":
        if bert_task == "ae":
            from berts.bat_ae import BertForABSA
            model = BertForABSA.from_pretrained(bert_model, num_labels = len(label_list), dropout=c.ae_lap_dropout, epsilon=c.ae_lap_epsilon) 
        elif bert_task == "asc":  
            from berts.bat_asc import BertForABSA  
            model = BertForABSA.from_pretrained(bert_model, num_labels = len(label_list), dropout=c.asc_lap_dropout, epsilon=c.asc_lap_epsilon)   
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

