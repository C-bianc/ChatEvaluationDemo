import csv
import logging
import os
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


def get_logger(name=None):
    """logger with the caller's module name."""
    if name is None:
        import inspect

        frame = inspect.stack()[1]
        name = frame.frame.f_globals["__name__"]

    return logging.getLogger(name)



# TODO : output seed scores (also maybe need to store them in the eval message

def save_conversation_with_evaluation(conversation, conv_id=None):
    """Input: a list of MessageEvaluation objects.
    Output: a csv file with the conversation and evaluation results.
    """

    if not conv_id:
        conv_id = datetime.now().strftime("%Y%m%d%H%M%S")
    output_filename = f"conversation_{conv_id}.csv"

    project_root = Path(__file__).parent.parent.parent
    output_file = os.path.join(project_root, "logs", output_filename)
    file_exists = os.path.exists(output_file)

    headers = [
        "conv_id",
        "turn",
        "author",
        "message",
        "intent",
        "prob_intent",
        "elicitation",
        "prob_elicit",
        "helpfulness",
        "prob_helpfulness",
        'seed_total',
        'seed_intent',
        'seed_output',
        'seed_helpful',

    ]

    with open(output_file, "w", newline="") as output_file:
        csv_writer = csv.writer(output_file)
        if not file_exists:
            csv_writer.writerow(headers)

        for message_evaluation in conversation:
            row = [conv_id, message_evaluation.turn_number, message_evaluation.role, message_evaluation.content]

            # evaluation results is a list of 3 PredictionResult
            # the labels are in the order of the headers (['intent', 'elicitation', 'helpfulness'])
            if message_evaluation.role == "bot":
                evaluations = message_evaluation.evaluation
                seed_results = message_evaluation.seed_scores

                for evaluation in evaluations:
                    probability = evaluation.logits[evaluation.label]
                    row.extend([evaluation.label, probability])

                for seed_result in seed_results.values():
                    row.extend([seed_result])
            csv_writer.writerow(row)

    action_log = "updated to existing file" if file_exists else "saved to"
    logging.info(f"Conversation {action_log} '{output_filename}'")
