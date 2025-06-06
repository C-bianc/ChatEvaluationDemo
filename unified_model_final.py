#!/usr/bin/env python
# ~* coding: utf-8 *~
import logging
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
from safetensors.torch import load_file
from torch import Tensor
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.utils.constants import MODEL_NAME
from app.utils.model_input_formatting import format_input

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    dimension: str
    label: str
    logits: Dict[str, float]



class MultiTaskBert(nn.Module):
    def __init__(self):
        super(MultiTaskBert, self).__init__()

        self.model_name = MODEL_NAME
        self.label2id_config = {
            "Communicative_Intent": {"D": 0, "I": 1, "O": 2},
            "Output_Elicitation": {"No": 0, "Yes": 1},
            "Helpfulness": {"Helpful": 0, "Neutral": 1, "Not helpful": 2},
        }

        self.id2label_config= {dim: {v: k for k, v in self.label2id_config[dim].items()} for dim in self.label2id_config}
        self.dimensions_config = ["Communicative_Intent", "Output_Elicitation", "Helpfulness"]

        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden_size = self.model.config.hidden_size

        self.intent_classifier = nn.Linear(hidden_size, 3)
        self.elicitation_classifier = nn.Linear(hidden_size, 2)
        self.helpfulness_classifier = nn.Linear(hidden_size, 3)

    @classmethod
    def load_model_from_checkpoint(cls, checkpoint_path):

        instance = cls()
        logger.info(f"Loading model on device: {instance.device}")

        raw_state = load_file(checkpoint_path)
        infer_state = {k: v for k, v in raw_state.items() if not k.startswith("loss_fn_")}

        instance.load_state_dict(infer_state)
        instance = instance.to(instance.device)

        instance.eval()
        return instance

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        context_input_ids: Tensor,
        context_attention_mask: Tensor,
    ) -> Dict[str, Tensor]:

        single_turn_outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        single_turn_cls_embedding = single_turn_outputs.hidden_states[-1][:, 0, :]

        multi_turn_outputs = self.model(
            input_ids=context_input_ids, attention_mask=context_attention_mask, output_hidden_states=True
        )
        multi_turn_cls_embedding = multi_turn_outputs.hidden_states[-1][:, 0, :]

        intent_logits = self.intent_classifier(single_turn_cls_embedding)
        elicitation_logits = self.elicitation_classifier(single_turn_cls_embedding)
        helpfulness_logits = self.helpfulness_classifier(multi_turn_cls_embedding)

        return {
            "intent_logits": intent_logits,
            "elicitation_logits": elicitation_logits,
            "helpfulness_logits": helpfulness_logits,
        }

    def _tokenizer_func_unified(self, convo: List[Dict[str, str]], turn: str) -> Dict[str, Tensor]:
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
            "input_ids": tokenized_target["input_ids"].to(self.device),
            "attention_mask": tokenized_target["attention_mask"].to(self.device),
            "context_input_ids": tokenized_context["input_ids"].to(self.device),
            "context_attention_mask": tokenized_context["attention_mask"].to(self.device),
        }

    def predict(self, convo: List[Dict[str, str]], turn: str) -> Dict[str, Tensor]:
        inputs = self._tokenizer_func_unified(convo, turn)
        with torch.no_grad():
            outputs = self.forward(**inputs)
        return outputs

    def decode_outputs(self, predictions: Dict[str, torch.Tensor]) -> List[PredictionResult]:
        output_list: List[PredictionResult] = []

        for dim_name, logits in zip(self.dimensions_config, predictions.values()):
            probabilities = torch.softmax(logits, dim=-1).flatten().tolist()
            probabilities = {label:round(prob, 2) for label, prob in zip(self.label2id_config[dim_name].keys(), probabilities)}
            sorted_probabilities = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))

            pred_label_id = torch.argmax(logits, dim=-1).item()
            label = self.id2label_config[dim_name][pred_label_id]

            output_list.append(PredictionResult(dimension=dim_name, label=label, logits=sorted_probabilities))

        return output_list

