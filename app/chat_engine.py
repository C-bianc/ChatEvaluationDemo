import re
import textwrap
from datetime import datetime
from typing import List

import gradio as gr
import pandas as pd

from app.components.about_info_ui import add_info_about_app
from app.components.chat_ui import create_chat_module
from app.components.evaluation_ui import create_evaluation_module
from app.components.prompt_ui import create_prompt_for_refinement_preview
from app.components.prompt_ui import create_prompt_parameters_module, create_prompt_preview_module
from app.evaluator import ConversationEvaluator
from app.utils.constants import MODEL_PATH, LLMS, ACTION_INTENT, ACTION_OUTPUT, ACTION_HELPFUL
from app.utils.conversation import create_prompt_template, get_css, load_config, format_seed_results
from app.utils.conversation import format_highlight_evaluation_results
from app.utils.logger import get_logger, save_conversation_with_evaluation
from compute_seed import Seed
from unified_model_final import MultiTaskBert, PredictionResult

logger = get_logger(__name__)

bert_evaluation_model = MultiTaskBert.load_model_from_checkpoint(MODEL_PATH)
evaluator = ConversationEvaluator(bert_evaluation_model)
seed = Seed()  # can put weights


def query_llm(model_choice, prompt):
    from ollama import generate

    response = generate(model=model_choice, prompt=prompt)
    return response.response


def get_prompt_for_refinement(labels, bot_response, history):

    logger.info(f"refining: {bot_response}")
    refinement_instructions = f"This generated bot response was labeled as {" and, ".join(labels)}.\n"
    task = textwrap.dedent(f"""\nTask: generate a new bot response that meets the previous requirements but keeps the same meaning.
    The new bot response can have a different structure so that it meets the previous requirements.
    Output only the refined bot response and nothing else. Do not print explanations. Do not add special characters.\n
            Bot response: {bot_response}
""")
    refinement_instructions += f"The previous content was: {history}\n"

    if ACTION_HELPFUL in labels:
        refinement_instructions += textwrap.dedent(
            f"""
        If {ACTION_HELPFUL}, the bot response should be more helpful:\n
        A helpful response is considered as helpful if:
        - it conveys one idea at a time, 
        - uses simple and easy to understand language
        - is not ambiguous (direct language)\n
    """
        )

    if ACTION_OUTPUT in labels:
        refinement_instructions += f"If {ACTION_OUTPUT}:\nThe bot response is not eliciting output. Keep the original meaning of the bot response in brackets and append to it a question or an imperative statement to the end of the response that would be a great continuity to the current content.\n"

    if ACTION_INTENT in labels:
        refinement_instructions += f"If {ACTION_INTENT}:\nThe bot response lacks describing someone or introducing someone as intent. Add a statement or question to expose to these intents.\n"

    refinement_instructions += task
    return refinement_instructions


def end_conversation_prompt(history):
    return f"""
    Given this conversation in a role-play game: 
    \nHistory: \n
        {history}\n
    
    Give a response as that ends the conversation with the player. You are the bot and are a NPC. 
    Output only the response and nothing else. Do not print explanations."""


def generate_bot_response(history, model_choice, context, requirements):
    structured_prompt = create_prompt_template(context, requirements, history)

    bot_response = query_llm(model_choice, structured_prompt).strip('"')  # we need user input for generating a response
    bot_response = re.sub(r'\[(.*?)\]', r'\1', bot_response)  # remove brackets
    bot_response = re.sub(r'\*\*(.*?)\*\*', r'\1', bot_response)  # remove bold markdown


    return bot_response


def get_actions_for_refinement(
    evaluation: List[PredictionResult], subscore_intent: float, subscore_output: float, subscore_helpfulness: float
):
    helpfulness_label = evaluation[2].label  # [dim1, dim2, dim3]
    actions_for_refinement = []

    if helpfulness_label == "Not helpful" or subscore_helpfulness < 0.25:
        actions_for_refinement.append(ACTION_HELPFUL)
        gr.Warning(f"Bot response was flagged not helpful. Generating another bot response...")

    if subscore_output < 0.50:
        actions_for_refinement.append(ACTION_OUTPUT)
        gr.Warning(f"Bot response was flagged not eliciting output. Generating another bot response...")

    if subscore_intent < 0.50:
        actions_for_refinement.append(ACTION_INTENT)
        gr.Warning(f"The user is not practicing enough the learning goals. Generating another bot response...")
    return actions_for_refinement


def get_new_evaluation(history, new_bot_response, user_replies):

    evaluation_of_refined_response = evaluator.evaluate_turn(convo=history, turn=new_bot_response)
    evaluator.add_message_evaluation(turn=new_bot_response, author="bot", evaluation_results=evaluation_of_refined_response)

    seed_total = seed.compute_total_seed(
        evaluator.intent_labels, evaluator.output_labels, evaluator.helpfulness_labels, user_replies
    )

    return evaluation_of_refined_response, seed_total


def chat_and_evaluate(user_input, conversation_history, model_choice, context, requirements):
    """
    1) update history with recent user input
    2) get bot response
    3) evaluate bot response and compute seed
    4) check if response needs to be refined + update last evaluation with new
    5) if seed is high and conversation is long enough, end conversation
    """
    # since history is only updated afterwards, we need to add manually for our query evaluation
    user_prompt = {"role": "user", "content": user_input}
    evaluator.add_message_evaluation(turn=user_input, author="user")
    conversation_history.append(user_prompt)
    user_replies = [msg["content"] for msg in conversation_history if msg["role"] == "user"]

    ### BOT RESPONSE
    bot_response = generate_bot_response(conversation_history, model_choice, context, requirements)

    #### formatted history
    formatted_history_string = evaluator.get_all_text_messages()  # Bot: msg, User: msg, ....

    logger.info(
        f"""
    \nHistory: \n
        {formatted_history_string}\n 
    Bot response: \n
        {bot_response}\n
    Model: {model_choice}
    """
    )

    ### EVALUATION
    evaluation: List[PredictionResult] = evaluator.evaluate_turn(convo=conversation_history, turn=bot_response)
    evaluator.add_message_evaluation(turn=bot_response, author="bot", evaluation_results=evaluation)
    logger.info(evaluator.get_conversation_evaluation())

    #### SEED
    seed_results = seed.compute_total_seed(
        evaluator.intent_labels, evaluator.output_labels, evaluator.helpfulness_labels, user_replies
    ) # is a dict containing all the scores
    seed_total = seed_results["seed"]
    logger.info(f"Seed total: {seed_total}")
    evaluator.update_last_message_seed_scores(seed_results)

    ### Check if need to refine
    actions_for_refinement = get_actions_for_refinement(
        evaluation, seed.intent_subscore, seed.output_elicitation_subscore, seed.helpfulness_subscore
    )

    # IF REFINEMENT NEEDED
    if len(actions_for_refinement) > 0:
        bad_response = evaluator.evaluated_conversation[-1]
        bad_response.reason_for_bad = ", ".join(actions_for_refinement)
        evaluator.add_bad_evaluation(bad_response)
        evaluator.remove_last_message_evaluation()

        bot_response = query_llm(
            model_choice, get_prompt_for_refinement(actions_for_refinement, bot_response, formatted_history_string)
        ).strip('"')

        ## update the evaluations with the newly generated bot response (need to do this because seed computes across all turns)
        evaluation, seed_results = get_new_evaluation(conversation_history, bot_response, user_replies)
        evaluator.update_last_message_seed_scores(seed_results)

    ### End conversation if seed is high and conversation is long enough
    if seed_total >= 0.70 and len(conversation_history) >= 20:
        bot_response = query_llm(model_choice, end_conversation_prompt(formatted_history_string)).strip('"')
        logger.info(f"Ending conversation: {bot_response}")

    conversation_history.pop()

    return (
        bot_response,
        format_highlight_evaluation_results(evaluation),
        format_seed_results(
            seed_results["seed"], seed_results["seed_intent"],seed_results["seed_output"], seed_results["seed_helpful"]
        ),
    )


def trigger_notification(msg):
    return gr.Info(msg, duration=3)

def build_chat_ui():
    with gr.Blocks(theme="ocean") as demo:
        gr.set_static_paths=["assets/"]
        default_context, default_requirements = load_config()
        conv_id = datetime.now().strftime("%Y%m%d%H%M%S")
        gr.HTML(f"<style>{get_css(__file__)}</style>")

        evaluation_box, seed_box = create_evaluation_module()

        with gr.Tab("Chat"):

            with gr.Row():
                ### LEFT COLUMN ### Prompt Configuration
                with gr.Column():
                    gr.Markdown("### Prefix Configuration")

                    with gr.Column(elem_id="prompt-column"):  # Main layout with 3 columns
                        prompt_preview = create_prompt_preview_module(default_context, default_requirements)
                        with gr.Column(scale=1, min_width=250):
                            context_input, requirements_input = create_prompt_parameters_module(
                                default_context, default_requirements, prompt_preview
                            )
                            prompt_preview.change(fn=lambda: trigger_notification("Prompt template updated."))
                            prompt_preview.render()

                ### MIDDLE COLUMN ### Chat Interface
                with gr.Column(scale=2, min_width=300):

                    with gr.Row(elem_classes="right-aligned-div"):
                        gr.Markdown("### Chat Interface")
                        save_conversation_button = gr.Button(
                            value="Save Conversation", variant="primary", size="sm", elem_id="save-conversation-button"
                        )

                    model_dropdown = gr.Dropdown(choices=LLMS, value=LLMS[0], label="Select a Model", render=False)
                    chatbot, chat_interface = create_chat_module(
                        chat_and_evaluate, model_dropdown, context_input, requirements_input, evaluation_box, seed_box
                    )
                    chatbot.clear(fn=evaluator.reset_conversation_evaluation)
                    chatbot.retry(fn=evaluator.remove_last_two_messages_evaluation)

                    save_conversation_button.click(
                        fn=lambda: (
                            save_conversation_with_evaluation(evaluator.get_evaluation_dataframe_with_bad_responses(), conv_id),
                            trigger_notification(f"Saved in logs/conversation_{conv_id}.csv"),
                        ),
                    )

                ### RIGHT COLUMN ### Evaluation
                with gr.Column(scale=1, min_width=250):
                    gr.Markdown("### Evaluation Results")
                    evaluation_box.render()

                    seed_box.render()
        with gr.Tab("Evaluation History"):
            gr.Markdown("### Evaluation History")
            with gr.Row():
                eval_history_box = gr.Dataframe(
                    wrap=True,    # ID # author # message # labels # seed # reason_for_bad
                    column_widths=[30, 50, 250, 250, 150, 200],
                    visible=True, value=pd.DataFrame(columns=["ID", "Author", "Message", "Labels", "Seed", "Reason for bad"]))
                chatbot.change(fn=lambda : evaluator.get_evaluation_dataframe_with_bad_responses(), outputs=eval_history_box, trigger_mode="always_last")

        with gr.Tab("Prompt Template"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Prompt Configuration")
                    prompt_preview_tab = create_prompt_preview_module(default_context, default_requirements)
                    context_input, requirements_input = create_prompt_parameters_module(
                        default_context, default_requirements, prompt_preview_tab
                    )
                with gr.Column():
                    gr.Markdown("### Prompt Preview")

                    prompt_preview_tab.label = "This prompt template is called at each turn for generating a response."
                    prompt_preview_tab.change(fn=lambda: trigger_notification("Prompt template updated."))
                    prompt_preview_tab.render()

            gr.Markdown("### Bot response refinement config")
            refinement_prompt = create_prompt_for_refinement_preview()
            refinement_prompt.value = get_prompt_for_refinement([ACTION_INTENT, ACTION_OUTPUT, ACTION_HELPFUL], "{bot response}", "{history}")

        with gr.Tab("About"):
            gr.Markdown(add_info_about_app())



    return demo
