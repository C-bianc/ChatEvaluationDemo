from dataclasses import dataclass, asdict
from typing import List

from app.unified_model_final import MultiTaskBert, PredictionResult
from app.utils.conversation import format_conversation


@dataclass
class MessageEvaluation:
    turn_number: int
    content: str
    role: str
    evaluation: List[PredictionResult]


class ConversationEvaluator:

    def __init__(self, evaluation_model: MultiTaskBert):
        self.model = evaluation_model
        self.evaluated_conversation: List[MessageEvaluation] = []
        self.intent_labels = []
        self.output_labels = []
        self.helpfulness_labels = []
        self.turn_number = 1

    def evaluate_turn(self, convo, turn) -> List[PredictionResult] :
        model_output = self.model.predict(convo, turn)
        labels = self.model.decode_outputs(model_output)
        return labels

    def add_message_evaluation(self, turn, author, evaluation_results):
        if author == "bot":
            self.intent_labels.append(evaluation_results[0].label)
            self.output_labels.append(evaluation_results[1].label)
            self.helpfulness_labels.append(evaluation_results[2].label) # evaluation results are always in this order

        self.evaluated_conversation.append(
            MessageEvaluation(turn_number=self.turn_number, content=turn, role=author, evaluation=evaluation_results)
        )
        self.turn_number += 1

    def get_conversation_evaluation(self):
        return self.evaluated_conversation

    def get_all_messages(self):
        conv_info_dict = [asdict(message_object) for message_object in self.evaluated_conversation]
        return format_conversation(conv_info_dict)

    def reset_conversation_evaluation(self):
        self.evaluated_conversation = []
        self.turn_number = 0

    def remove_last_message_evaluation(self):
        self.evaluated_conversation.pop()
        self.turn_number -= 1
