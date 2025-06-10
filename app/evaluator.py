from dataclasses import dataclass, asdict
from typing import List, Optional

import pandas as pd

from app.utils.conversation import format_conversation
from unified_model_final import MultiTaskBert, PredictionResult


@dataclass
class MessageEvaluation:
    turn_number: int
    content: str
    role: str
    evaluation: List[PredictionResult]
    seed_scores : Optional[dict]
    reason_for_bad: Optional[str]


class ConversationEvaluator:

    def __init__(self, evaluation_model: MultiTaskBert):
        self.model = evaluation_model
        self.evaluated_conversation: List[MessageEvaluation] = []
        self.intent_labels = []
        self.output_labels = []
        self.helpfulness_labels = []

        self.bad_evaluations: List[MessageEvaluation] = []
        self.turn_number = 1

    def evaluate_turn(self, convo, turn) -> List[PredictionResult]:
        model_output = self.model.predict(convo, turn)
        labels = self.model.decode_outputs(model_output)
        return labels

    def add_message_evaluation(self, turn, author, evaluation_results = None, seed_results = None, reason_for_bad = None):
        if author == "bot":
            self.intent_labels.append(evaluation_results[0].label)
            self.output_labels.append(evaluation_results[1].label)
            self.helpfulness_labels.append(evaluation_results[2].label)  # evaluation results are always in this order

        self.evaluated_conversation.append(
            MessageEvaluation(
                turn_number=self.turn_number,
                content=turn,
                role=author,
                evaluation=evaluation_results,
                seed_scores=seed_results,
                reason_for_bad=reason_for_bad
            )
        )
        self.turn_number += 1

    def get_conversation_evaluation(self):
        return self.evaluated_conversation

    def get_conversation_evaluation_with_bad_responses(self):
        merged_evaluations = self.get_conversation_evaluation().copy()
        # we want the current history, with the bot responses that were refined at the right index
        for bad_evaluation in self.bad_evaluations:
            merged_evaluations.insert(bad_evaluation.turn_number - 1, bad_evaluation)

        return merged_evaluations

    def get_all_text_messages(self):
        """
        returns a string with

        User:
        Bot:

        structure
        """
        conv_info_dict = [asdict(message_object) for message_object in self.evaluated_conversation]
        return format_conversation(conv_info_dict)

    def update_last_message_seed_scores(self, seed_scores: dict):
        """
        set the seed_scores for the most recently added message evaluation
        """
        if self.evaluated_conversation:
            self.evaluated_conversation[-1].seed_scores = seed_scores
        else:
            pass

    def add_bad_evaluation(self, evaluation: MessageEvaluation):
        self.bad_evaluations.append(evaluation)

    def reset_conversation_evaluation(self):
        self.evaluated_conversation = []
        self.bad_evaluations = []
        self.intent_labels = []
        self.output_labels = []
        self.helpfulness_labels = []
        self.turn_number = 1

    def remove_last_message_evaluation(self):
        self.evaluated_conversation.pop()
        self.turn_number -= 1

    def remove_last_two_messages_evaluation(self):
        """this is useful when user retries the reply of the bot.
        Gradio re-appends both the previous bot and the user message.
        """
        self.evaluated_conversation.pop()
        self.evaluated_conversation.pop()
        self.turn_number -= 2

    def get_evaluation_dataframe_with_bad_responses(self):

        all_history = self.get_conversation_evaluation_with_bad_responses()

        # Create lists to hold each column's data
        turn_numbers = []
        authors = []
        messages = []
        evaluations = []
        reasons = []
        seed_results = []

        for message_evaluation in all_history:
            turn_numbers.append(message_evaluation.turn_number)
            authors.append(message_evaluation.role)
            messages.append(message_evaluation.content)

            # format labels : label (prob)
            if message_evaluation.role == "bot":
                eval_text = ""
                for dim_eval in message_evaluation.evaluation:
                    eval_text += f"{dim_eval.label} ({dim_eval.logits[dim_eval.label]:.2f})\n"
                evaluations.append(eval_text)

                if message_evaluation.reason_for_bad:
                    reasons.append(message_evaluation.reason_for_bad)
                else:
                    reasons.append("Good response")

                seed_text = ""
                for seed_result in message_evaluation.seed_scores:
                    seed_text += f"{seed_result}: {message_evaluation.seed_scores[seed_result]:.2f}\n"
                seed_results.append(seed_text)
            else:
                evaluations.append("")
                seed_results.append("")
                reasons.append("")

        df = pd.DataFrame(
            {
                "ID": turn_numbers,
                "Author": authors,
                "Message": messages,
                "Evaluation": evaluations,
                "Seed": seed_results,
                "Reason for refinement": reasons,
            }
        )
        print(df)

        return df
