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


def save_conversation_with_evaluation(conversation_df, conv_id=None):
    """Input: a list of MessageEvaluation objects.
    Output: a csv file with the conversation and evaluation results.
    """

    if not conv_id:
        conv_id = datetime.now().strftime("%Y%m%d%H%M%S")
    output_filename = f"conversation_{conv_id}.csv"

    project_root = Path(__file__).parent.parent.parent
    output_file = os.path.join(project_root, "logs", output_filename)
    file_exists = os.path.exists(output_file)

    conversation_df = conversation_df.assign(conv_id=conv_id)

    conversation_df.to_csv(output_file, sep=";", index=False, mode="a" if file_exists else "w")

    action_log = "updated to existing file" if file_exists else "saved to"
    logging.info(f"Conversation {action_log} '{output_filename}'")
