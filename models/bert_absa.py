import torch
import os
import unicodedata
from label_studio_ml.model import LabelStudioMLBase
from absa_data_utils import Prediction, ABSATokenizer, InputExample
import absa_data_utils as data_utils
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import modelconfig
 
dir_path = os.path.dirname(os.path.realpath(__file__))

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

class BertABSA(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(BertABSA, self).__init__(**kwargs)
        self.ae_processor = data_utils.AeProcessor()
        self.ae_label_list = self.ae_processor.get_labels()
        self.asc_processor = data_utils.AscProcessor()
        self.asc_label_list = self.asc_processor.get_labels()
        self.domain = modelconfig.pick_domain()
        self.bert = dir_path + "/" + modelconfig.MODEL_ARCHIVE_MAP[self.domain +"_pt"]
        self.tokenizer = ABSATokenizer.from_pretrained(self.bert)
        self.bert_tasks = ["ae", "asc"]
        self.ae_model = modelconfig.pick_model(self.bert_tasks[0], self.bert, self.ae_label_list)
        self.asc_model = modelconfig.pick_model(self.bert_tasks[1], self.bert, self.asc_label_list)
        self.max_seq_length = 100
        self.batch_size = 3

    def aspectExtraction(self, texts):
        tokenized_tmp = []
        prediction_tmp = []
        lines = []
        for ids in range(len(texts)):
            guid = "%s-%s" % ("pre", ids)
            text_tokenized = self.tokenizer.tokenize(texts[ids])
            tokenized_tmp.append(text_tokenized)
            dummy_labels=['O' for i in range(len(text_tokenized))]
            lines.append(
                InputExample(guid = guid, text_a=text_tokenized, label = dummy_labels))

        lines_features = data_utils.convert_examples_to_features(lines, self.ae_label_list, self.max_seq_length, self.tokenizer, self.bert_tasks[0])
        all_input_ids = torch.tensor([f.input_ids for f in lines_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in lines_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in lines_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in lines_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.batch_size)
        self.ae_model.to(device)
        self.ae_model.eval()

        n_line = 0

        for step, batch in enumerate(eval_dataloader):
            i = 0
            batch = tuple(t.to(device) for t in batch)
            input_ids, segment_ids, input_mask, label_ids = batch
            
            with torch.no_grad():
                logits = self.ae_model(input_ids, segment_ids, input_mask)
                argmax_logits = logits.softmax(2).max(2)[1]
                max_logits = logits.softmax(2).max(2)[0]

            for l in range(len(batch[i])):
                tokenized_line = lines[n_line].text_a
                labels = argmax_logits[l][1:len(tokenized_line)+1]
                masks = label_ids[l][1:len(tokenized_line)+1] 
                scores = max_logits[l][1:len(tokenized_line)+1] 
                prediction_tmp.append(
                    Prediction(label = labels, mask = masks, score = scores, tokenized_line = tokenized_line)
                    )
                n_line = n_line + 1
            i = i + 1

        return prediction_tmp

    def aspectSentimentClassification(self, sentences, terms, texts):
        predictions = []
        labels = []
        lines = []

        for ids in range(len(sentences)):
            guid = "%s-%s" % ("pre", ids)
            label = self.asc_label_list[2]
            lines.append(
                InputExample(guid = guid, text_a = terms[ids], text_b = sentences[ids] , label = label))

        lines_features = data_utils.convert_examples_to_features(lines, self.asc_label_list, self.max_seq_length, self.tokenizer, self.bert_tasks[1])

        all_input_ids = torch.tensor([f.input_ids for f in lines_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in lines_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in lines_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in lines_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.batch_size)

        self.asc_model.cuda()
        self.asc_model.eval()

        for step, batch in enumerate(eval_dataloader):
            i = 0
            batch = tuple(t.cuda() for t in batch)

            input_ids, segment_ids, input_mask, label_ids = batch
            
            with torch.no_grad():
                logits = self.asc_model(input_ids, segment_ids, input_mask)

                argmax_logits = logits.softmax(1).max(1)[1]
                max_logits = logits.softmax(1).max(1)[0]

            for l in range(len(batch[i])):
                labels.append(argmax_logits[l].item())
            i = i + 1

        predictions.append(sentences)
        predictions.append(terms)
        predictions.append(labels)
        return predictions
    

    def predict(self, tasks, **kwargs):
        predictions = []
        predictions_ae = []
        predictions_asc = []
        sentences_asc = []
        terms_asc = []

        from_name, schema = list(self.parsed_label_config.items())[0]
        to_name = schema['to_name'][0]

        texts = [task['data']['ner'] for task in tasks]

        predictions_ae = self.aspectExtraction(texts)


        for n_line in range(len(texts)):
            terms = []
            sentence = texts[n_line]

            for n_word in range(len(predictions_ae[n_line].tokenized_line)):#per ogni parola tokenizzata
                if predictions_ae[n_line].label[n_word] == 1:#se il label è uguale a 1 ('B')
                    term_tmp = predictions_ae[n_line].tokenized_line[n_word]#salvo il termine
                    n_word = n_word + 1#vado al token successivo
                    while n_word<len(predictions_ae[n_line].tokenized_line) and predictions_ae[n_line].label[n_word]==2: #quando c è un punto anziche uno spazio non viene riconosciuto, cerco nella frase gia qui?
                        new_term = ' ' + predictions_ae[n_line].tokenized_line[n_word]
                        if(new_term.startswith(" ##")): 
                            new_term = new_term[3:]
                        term_tmp = term_tmp + new_term
                        n_word = n_word + 1
                    
                    terms.append(term_tmp)
            if len(terms)>0:
                for term in terms:
                    sentences_asc.append(sentence)
                    terms_asc.append(term)
            else: 
                sentences_asc.append(sentence)
                terms_asc.append('')
            
            print("***text: ", sentence, " - ", str(terms))

        predictions_asc = self.aspectSentimentClassification(sentences_asc, terms_asc, texts)

        for text in texts:
            print(text)
            prediction = []
            for i in range(len(predictions_asc[0])):
                sentence_f = predictions_asc[0][i]
                if(text == sentence_f):
                    sentence_f = unicodedata.normalize('NFKD', sentence_f.lower()).encode('ascii', 'ignore').decode('utf8')
                    term_f = predictions_asc[1][i]
                    label_f = predictions_asc[2][i]
                    if(sentence_f.find(term_f)!= -1 and term_f != ''):
                        start = sentence_f.find(term_f)
                        end = start + len(term_f)
                        print(term_f, start, end, label_f)
                        prediction.append({
                            'from_name': from_name,
                            'to_name': to_name,
                            'type': 'labels',
                            'value': {
                                'labels': [self.asc_label_list[label_f]],
                                'start': start,
                                'end': end, 
                                'text': term_f}
                                })
            predictions.append({
                'result': prediction,
            })
        print(predictions)
        return predictions
