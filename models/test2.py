import torch
import os
from label_studio_ml.model import LabelStudioMLBase

import ipdb

import absa_data_utils as data_utils
from absa_data_utils import ABSATokenizer
from bpsum_ae import BertForABSA
from absa_data_utils import InputExample
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import utils

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

class MyModel(LabelStudioMLBase):
    def __init__(self, **kwargs):
        # don't forget to initialize base class...
        super(MyModel, self).__init__(**kwargs)
        self.processor = data_utils.AeProcessor()
        self.label_list = self.processor.get_labels()
        self.bert = 'H:\git\label-studio-BERT-for-ABSA\my-ml-backend\pt_model\laptop_pt'
        self.tokenizer = ABSATokenizer.from_pretrained(self.bert)
        self.model = BertForABSA.from_pretrained(self.bert, num_labels = len(self.label_list))
        self.max_seq_length = 100
        self.bert_task = "ae"
        self.batch_size = 3

    def predict(self, tasks, **kwargs):
        predictions = []
        # Get annotation tag first, and extract from_name/to_name keys from the labeling config to make predictions
        from_name, schema = list(self.parsed_label_config.items())[0]
        to_name = schema['to_name'][0]
    
        for task in tasks:
            # es task
            # {'id': 28, 'data': {'text': 'le scarpe sono utili'},
            # 'meta': {}, 'created_at': '2021-07-20T14:44:54.023502Z',
            # 'updated_at': '2021-07-20T14:44:54.023502Z', 'is_labeled': False, 
            # 'overlap': 1, 'project': 4, 'file_upload': 22, 'annotations': [], 'predictions': []}

            #text = task['data']['text']
            texts = [task['data']['text'] for task in tasks]
            
            lines = []
            for ids in range(len(texts)):
                guid = "%s-%s" % ("test", ids )
                label = ['O' for i in texts[ids]]
                lines.append(
                    InputExample(guid = guid, text_a=texts[ids], label = label))

            lines_features = data_utils.convert_examples_to_features(lines, self.label_list, self.max_seq_length, self.tokenizer, "ae")

            all_input_ids = torch.tensor([f.input_ids for f in lines_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in lines_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in lines_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in lines_features], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)

            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.batch_size)

            self.model.to(device)
            self.model.eval()

            for step, batch in enumerate(eval_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, segment_ids, input_mask, label_ids = batch
                
                with torch.no_grad():
                    logits = self.model(input_ids, segment_ids, input_mask)


            ipdb.set_trace()

            ##############
            predictions.append({
                'result': [{
                    'from_name': from_name,
                    'to_name': to_name,
                    'type': 'choices',
                    'value': {'choices': ['Negative']}
                }]
            })
        return predictions


