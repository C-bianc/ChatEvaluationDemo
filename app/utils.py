import json

from constants import CONFIG_FILE


def add_sep(text, tokenizer, is_cls=False):
    if not text:
        return ""
    if is_cls:
        return tokenizer.cls_token + text
    bos = tokenizer.sep_token if tokenizer.bos_token is None else tokenizer.bos_token
    return f"{bos}{text}{tokenizer.sep_token}"


def format_input(convo, target, tokenizer):
    n_turns  = len(convo)
    convo = convo[-1: -3]
    print(convo)

    prev_bot_text = convo[n_turns - 2]["content"] if n_turns == 2 else ""
    user_text = convo[n_turns - 1]["content"] if n_turns == 1 else ""

    prev_bot_text = add_sep(prev_bot_text, tokenizer, True)
    user_text = add_sep(user_text, tokenizer)

    if not prev_bot_text:
        user_text = add_sep(user_text, tokenizer, True)

    context = prev_bot_text + user_text

    if not context:
        target = add_sep(target, tokenizer, True) + tokenizer.sep_token
    else:
        target = add_sep(target, tokenizer)

    text = context + target
    print(text)
    return text


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
    prefix = f"""
Given this conversation settings:
{context}
    
Given this conversation history:
{format_conversation(history) if history else "{a history}"}
    
Provide an answer that meets the following requirements:
{requirements}
    """
    return prefix


def join_with_newline(input):
    return "\n".join(input)


def load_config():
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            return config.get("context"), config.get("requirements")
    except Exception as e:
        print(e)
        pass
        return None, None


def save_config_and_preview(context, requirements):
    saved_context, saved_requirements = load_config()

    if context is None:
        context = saved_context
    if requirements is None:
        requirements = saved_requirements


    with open(CONFIG_FILE, "w") as f:
        json.dump({"context": context, "requirements": requirements}, f)

    return "Prefix prompt updated.", create_prompt_template(
        context, requirements, "{history}"
    )


def save_context_config(context):
    return save_config_and_preview(context, None)


def save_requirements_config(requirements):
    return save_config_and_preview(None, requirements)
