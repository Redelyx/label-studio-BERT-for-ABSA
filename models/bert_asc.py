import torch
import os
from label_studio_ml.model import LabelStudioMLBase
import logging

import absa_data_utils as data_utils
from absa_data_utils import ABSATokenizer
from absa_data_utils import InputExample
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import pick_bert
import modelconfig
 
dir_path = os.path.dirname(os.path.realpath(__file__))

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


class Prediction(object):
    def __init__(self, label, mask, score):
        self.label = label
        self.mask = mask
        self.score = score

class BertASC(LabelStudioMLBase):
    def __init__(self, **kwargs):
        # don't forget to initialize base class...
        super(BertASC, self).__init__(**kwargs)
        self.processor = data_utils.AscProcessor()
        self.label_list = self.processor.get_labels()
        self.domain = pick_bert.pick_domain()
        self.bert = dir_path + "/" + modelconfig.MODEL_ARCHIVE_MAP[self.domain]
        self.tokenizer = ABSATokenizer.from_pretrained(self.bert)
        self.bert_task = "asc"
        self.model = pick_bert.pick_model(self.bert_task, self.bert, self.label_list)
        self.max_seq_length = 100
        self.batch_size = 3

    def predict(self, tasks, **kwargs):
        predictions = []
        prediction_tmp = []
        lines = []
        # Get annotation tag first, and extract from_name/to_name keys from the labeling config to make predictions
        from_name, schema = list(self.parsed_label_config.items())[0]
        to_name = schema['to_name'][0]

        sentences = [task['data']['sentence'] for task in tasks]
        terms = [task['data']['term'] for task in tasks]

        for ids in range(len(tasks)):
            guid = "%s-%s" % ("pre", ids)
            label = self.label_list[2]
            lines.append(
                InputExample(guid = guid, text_a = terms[ids], text_b = sentences[ids] , label = label))

        lines_features = data_utils.convert_examples_to_features(lines, self.label_list, self.max_seq_length, self.tokenizer, self.bert_task)

        all_input_ids = torch.tensor([f.input_ids for f in lines_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in lines_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in lines_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in lines_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.batch_size)

        self.model.cuda()
        self.model.eval()

        for step, batch in enumerate(eval_dataloader):
            i = 0
            batch = tuple(t.cuda() for t in batch)

            input_ids, segment_ids, input_mask, label_ids = batch
            
            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)

                argmax_logits = logits.softmax(1).max(1)[1]
                max_logits = logits.softmax(1).max(1)[0]
            for l in range(len(batch[i])):
                prediction_tmp.append(
                    Prediction(label = argmax_logits[l], mask = label_ids[l], score = max_logits[l]))
            i = i + 1

        for n_line in range(len(sentences)):
            prediction = []
            print("***line: ", n_line, " - ", sentences[n_line], "   Aspect: ", terms[n_line])
            sentence = sentences[n_line]
            term = terms[n_line]
            if(sentence.find(term)!= -1):
                start = sentence.find(term)
                end = start + len(term)
            label = prediction_tmp[n_line].label
            score = prediction_tmp[n_line].score.item()
            
            print(start, end, label)
            predictions.append({
                'result':[{
                    'from_name': from_name,
                    'to_name': to_name,
                    'type': 'labels',
                    'value': {
                        'labels': [self.label_list[label]],
                        'start': start,
                        'end': end, 
                        'text': term}
                }],
                'score': score
            })

        return predictions
