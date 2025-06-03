from typing import Dict, List

from app.utils.logger import get_logger

logger = get_logger(__name__)


def add_sep(text, tokenizer, is_cls=False):
    if not text:
        return ""
    if is_cls:
        return tokenizer.cls_token + text
    bos = tokenizer.sep_token if tokenizer.bos_token is None else tokenizer.bos_token
    return f"{bos}{text}{tokenizer.sep_token}"


def format_input(convo: List[Dict[str, str]], target: str, tokenizer) -> str:
    convo = convo[-2:]
    n_turns = len(convo)

    prev_bot_text = convo[-2]["content"] if n_turns == 2 else ""
    user_text = convo[-1]["content"] if n_turns >= 1 else ""

    prev_bot_text = add_sep(prev_bot_text, tokenizer, True)

    if not prev_bot_text and user_text:
        user_text = add_sep(user_text, tokenizer, True) + tokenizer.sep_token
    else:
        user_text = add_sep(user_text, tokenizer)

    context = prev_bot_text + user_text

    if not context:
        target = add_sep(target, tokenizer, True) + tokenizer.sep_token

    else:
        target = target + tokenizer.sep_token

    text = context + target
    logger.info(f"Formatted input fed to the model: {text}")
    return text
