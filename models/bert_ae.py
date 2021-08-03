import torch
import os
from label_studio_ml.model import LabelStudioMLBase
import absa_data_utils as data_utils
from absa_data_utils import ABSATokenizer, Prediction, InputExample
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import pick_bert
import modelconfig
 
dir_path = os.path.dirname(os.path.realpath(__file__))

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
class BertAE(LabelStudioMLBase):
    def __init__(self, **kwargs):
        # don't forget to initialize base class...
        super(BertAE, self).__init__(**kwargs)
        self.processor = data_utils.AeProcessor()
        self.label_list = self.processor.get_labels()
        self.domain = pick_bert.pick_domain()
        self.bert = dir_path + "/" + modelconfig.MODEL_ARCHIVE_MAP[self.domain]
        self.tokenizer = ABSATokenizer.from_pretrained(self.bert)
        self.bert_task = "ae"
        self.model = pick_bert.pick_model(self.bert_task, self.bert, self.label_list)
        self.max_seq_length = 100
        self.batch_size = 3

    def predict(self, tasks, **kwargs):
        predictions = []
        prediction_tmp = []
        tokenized_tmp = []
        lines = []
        # Get annotation tag first, and extract from_name/to_name keys from the labeling config to make predictions
        from_name, schema = list(self.parsed_label_config.items())[0]
        to_name = schema['to_name'][0]

        texts = [task['data']['text'] for task in tasks]

        for ids in range(len(tasks)):
            guid = "%s-%s" % ("pre", ids)
            text_tokenized = self.tokenizer.tokenize(texts[ids])
            tokenized_tmp.append(text_tokenized)
            label=['O' for i in range(len(text_tokenized))]
            lines.append(
                InputExample(guid = guid, text_a=text_tokenized, label = label))

        lines_features = data_utils.convert_examples_to_features(lines, self.label_list, self.max_seq_length, self.tokenizer, self.bert_task)

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
            i = 0
            batch = tuple(t.to(device) for t in batch)
            input_ids, segment_ids, input_mask, label_ids = batch
            
            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)

                argmax_logits = logits.softmax(2).max(2)[1]
                max_logits = logits.softmax(2).max(2)[0]
            for l in range(len(batch[i])):
                prediction_tmp.append(
                    Prediction(label = argmax_logits[l], mask = label_ids[l], score = max_logits[l]))

            i = i + 1

        for n_line in range(len(tokenized_tmp)):
            start = 0
            end = 0
            prediction = []
            print("***line: ", n_line, " - ", texts[n_line])
            for n_word in range(len(tokenized_tmp[n_line])):
                word = tokenized_tmp[n_line][n_word]
                label = prediction_tmp[n_line].label[n_word + 1]

                end = start + len(word) + 1
                
                if(label != 0):
                    print(word, start, end, label)
                    prediction.append({
                        'from_name': from_name,
                        'to_name': to_name,
                        'type': 'labels',
                        'value': {
                            'labels': [self.label_list[label]],
                            'start': start,
                            'end': end, 
                            'text': word}
                            })
                start = end

            predictions.append({
                'result': prediction,
            })
        return predictions
