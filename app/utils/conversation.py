import json
import os
from typing import List, Optional, Tuple

from app.utils.constants import CONFIG_FILE
from app.utils.logger import get_logger
from unified_model_final import PredictionResult

logger = get_logger(__name__)


def format_conversation(conversation):
    concatenated_conversation = ""
    for turn in conversation:
        message = turn["content"]
        author = turn["role"]

        if author == "user":
            concatenated_conversation += f"User: {message}\n"
        elif author == "assistant":
            concatenated_conversation += f"Bot: {message}\n"

    return concatenated_conversation


def create_prompt_template(context, requirements, history):
    prefix = f"""Given this conversation settings:\n
{context}
    
Given this conversation history:
{format_conversation(history) if history else "{a history}"}
    
Provide an answer that meets the following requirements:
{requirements}
    """
    return prefix


def join_with_newline(input):
    print(input)
    return "\n".join(input)


def load_config():
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            if "context" not in config or "requirements" not in config:
                logger.error("Missing prompt config. Please add missing keys to 'prompt_prefix.json'.")
                return None, None
    except Exception as e:
        logger.error("Failed to load configuration: %s", e)
        return None, None

    return config["context"], config["requirements"]


def save_config_and_preview(context, requirements):
    saved_context, saved_requirements = load_config()

    if context is None:
        context = saved_context
    if requirements is None:
        requirements = saved_requirements


    with open(CONFIG_FILE, "w") as f:
        json.dump({"context": context, "requirements": requirements}, f, ensure_ascii=False, indent=4)

    return create_prompt_template(
        context, requirements, None
    )


def save_context_config(context):
    return save_config_and_preview(context, None)


def save_requirements_config(requirements):
    return save_config_and_preview(None, requirements)

def get_css(current_file):
    current_dir = os.path.dirname(os.path.abspath(current_file))
    css_file_path = os.path.join(current_dir,"components", "style.css")

    with open(css_file_path, "r") as file:
        css_content = file.read()
        return css_content


def format_result(label, probability, index):
    mapping = {0: "argmax", 1: "next_highest", 2: "third_highest"}
    segments = [
        (f"{label}", mapping[index]),
        (f"({probability:.2f})", None)
    ]

    return segments

def format_highlight_evaluation_results(evaluation_results: List[PredictionResult]) -> List[Tuple[str, Optional[str]]]:
    """
    since gradio HighlightedText needs segments of tuples (text, color)
    we need to format this way our PredictionResult objects
    """

    highlighted_segments = []

    for result in evaluation_results:
        # Add dimension
        highlighted_segments.append((f"{result.dimension.replace("_", " ")}\n", None))

        probabilities = result.logits  # {label : probability} sorted in descending order
        logger.info(f"Evaluation results: {probabilities}")
        for i, (label, probability) in enumerate(probabilities.items()):

            # add labels and probability
            if len(probabilities) == 2 and i == 1:
                highlighted_segments += format_result(label, probability, i + 1)
            else:
                highlighted_segments += format_result(label, probability, i)

        highlighted_segments.append(("\n\n", None))
    highlighted_segments.pop()

    return highlighted_segments


def format_seed_results(total, intent, output, helpfulness):
    if total > 0.5:
        total_color = "satisfying"

    elif total > 0.3:
        total_color = "enough"
    else:
        total_color = "not enough"

    return [
        (f"{total}\n", total_color),
        # intent
        (f"intent = {intent}\n", "subscore"),
        ("ratio of 'D' and 'I'\n", None),
        # output
        (f"elicitation = {output}\n", "subscore"),
        ("ratio of 'yes' + learner contribution (penalized according to avg resp len)\n", None),
        # helpfulness
        (f"helpfulness = {helpfulness}\n", "subscore"),
        ("ratio of helpful", None),
    ]
