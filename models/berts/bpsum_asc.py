import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertLayer
from transformers.models.bert.modeling_bert import BertPooler

class PSUM(nn.Module):
    def __init__(self, count, config, num_labels):
        super(PSUM, self).__init__()
        self.count = count
        self.num_labels = num_labels
        self.pooler = BertPooler(config)
        self.pre_layers = torch.nn.ModuleList()
        self.loss_fct = torch.nn.ModuleList()
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        for i in range(count):
            self.pre_layers.append(BertLayer(config))
            self.loss_fct.append(torch.nn.CrossEntropyLoss(ignore_index=-1))

    def forward(self, layers, attention_mask, labels):
        losses = []
        logitses = []
        for i in range(self.count):
            layer = self.pre_layers[i](layers[-i-1], attention_mask)
            layer = self.pooler(layer[0])
            logits = self.classifier(layer)
            if labels is not None:
                loss = self.loss_fct[i](logits.view(-1, self.num_labels), labels.view(-1))
                losses.append(loss)
            logitses.append(logits)
        if labels is not None:
            total_loss = torch.sum(torch.stack(losses), dim=0)
        else:
            total_loss = torch.Tensor(0)
        avg_logits = torch.sum(torch.stack(logitses), dim=0)/self.count
        return total_loss, avg_logits


class BertForABSA(BertPreTrainedModel):
    def __init__(self, config, num_labels=3):
        super(BertForABSA, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.psum = PSUM(4, config, num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        layers = self.bert(input_ids, token_type_ids = token_type_ids, 
                                                        attention_mask=attention_mask, 
                                                        output_hidden_states=True)['hidden_states']
        mask = self.bert.get_extended_attention_mask(attention_mask, input_ids.shape, device = layers[0].device)
        loss, logits = self.psum(torch.stack(layers)[1:], mask, labels)
        if labels is not None:
            return loss
        else:
            return logits