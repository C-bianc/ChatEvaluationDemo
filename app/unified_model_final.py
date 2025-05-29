#!/usr/bin/env python
# ~* coding: utf-8 *~
from collections import defaultdict

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from constants import MODEL_NAME, ID2LABEL, DIMENSIONS
from utils import format_input


# =========== MODEL


class MultiTaskBert(nn.Module):
    def __init__(self, model_name=MODEL_NAME, n_intent_dim=3, n_elicit_dim=2, n_helpful_dim=3):
        super(MultiTaskBert, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        hidden_size = self.model.config.hidden_size

        self.intent_classifier = nn.Linear(hidden_size, n_intent_dim)  # dim1
        self.elicitation_classifier = nn.Linear(hidden_size, n_elicit_dim)  # dim2
        self.helpfulness_classifier = nn.Linear(hidden_size, n_helpful_dim)  # dim3

    def forward(self, input_ids, attention_mask, context_input_ids, context_attention_mask, labels=None):
        single_turn_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        single_turn_cls_embedding = single_turn_outputs.hidden_states[-1][:, 0, :]

        multi_turn_outputs = self.model(
            input_ids=context_input_ids, attention_mask=context_attention_mask, output_hidden_states=True
        )
        multi_turn_cls_embedding = multi_turn_outputs.hidden_states[-1][:, 0, :]

        # ==== LOGITS ====
        intent_logits = self.intent_classifier(single_turn_cls_embedding)  # dim1
        elicitation_logits = self.elicitation_classifier(single_turn_cls_embedding)  # dim2
        helpfulness_logits = self.helpfulness_classifier(multi_turn_cls_embedding)  # dim3

        return {
            "intent_logits": intent_logits,
            "elicitation_logits": elicitation_logits,
            "helpfulness_logits": helpfulness_logits,
        }

    def _tokenizer_func_unified(self, convo, turn):
        print("turn", turn)
        print("convo", convo)
        tokenized_target = self.tokenizer(
            turn,
            truncation=True,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        tokenized_context = self.tokenizer(
            format_input(convo, turn, self.tokenizer),
            truncation=True,
            add_special_tokens=False,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": tokenized_target["input_ids"],
            "attention_mask": tokenized_target["attention_mask"],
            "context_input_ids": tokenized_context["input_ids"],
            "context_attention_mask": tokenized_context["attention_mask"],
        }


    def predict(self, convo, turn):
        inputs = self._tokenizer_func_unified(convo, turn)
        with torch.no_grad():
            outputs = self.forward(**inputs)
        return outputs


    @staticmethod
    def decode_outputs(predictions, return_dict=False):
        string_output = ""
        results = defaultdict()

        for dim, logits in zip(DIMENSIONS, predictions.values()):
            pred_label_id = torch.argmax(logits, dim=-1).item()
            label = ID2LABEL[dim][pred_label_id]

            string_output += f"{dim}: {label}\n"
            results[dim] = label

        return results if return_dict else string_output
