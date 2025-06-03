import re
from datetime import datetime
from typing import List

import gradio as gr

from app.components.chat_ui import create_chat_module
from app.components.evaluation_ui import create_evaluation_module
from app.components.prompt_ui import create_prompt_parameters_module, create_prompt_preview_module
from app.evaluator import ConversationEvaluator
from app.unified_model_final import MultiTaskBert, PredictionResult
from app.utils.constants import MODEL_PATH, LLMS
from app.utils.conversation import create_prompt_template, get_css, load_config, format_seed_results
from app.utils.conversation import format_highlight_evaluation_results
from app.utils.logger import get_logger, save_conversation_with_evaluation
from compute_seed import Seed

logger = get_logger(__name__)

bert_evaluation_model = MultiTaskBert.load_model_from_checkpoint(MODEL_PATH)
evaluator = ConversationEvaluator(bert_evaluation_model)
seed = Seed()  # can put weights


def query_llm(model_choice, prompt):
    from ollama import generate

    response = generate(model=model_choice, prompt=prompt)
    return response.response


def get_prompt_helpfulness_refinement(user_input, bot_response):
    logger.info(f"refining: {bot_response}")
    return f"""This generated bot response was labeled as not helpful. 
       A helpful response is considered as helpful if it conveys one idea at a time, 
       uses simple and not ambiguous language,
       or was not responsive to the user's input.
       
       Task: generate another bot response that meets the previous requirements. Output only the response and nothing else.
       User input: {user_input}
       Bot response: {bot_response} 
       """


def chat_and_evaluate(user_input, conversation_history, model_choice, context, requirements):
    # since history is only updated afterwards, we need to add manually for our query evaluation
    user_prompt = {"role": "user", "content": user_input}
    evaluator.add_message_evaluation(turn=user_input, author="user", evaluation_results=[])
    conversation_history.append(user_prompt)

    structured_prompt = create_prompt_template(context, requirements, conversation_history)

    ### BOT RESPONSE
    bot_response = query_llm(model_choice, structured_prompt).strip('"')  # we need user input for generating a response
    bot_response = re.sub(r"\*(.*?)\*", "", bot_response)  # avoid inline actions

    logger.info(
        f"""
    \nHistory: \n
        {evaluator.get_all_messages()}\n 
    Bot response: \n
        {bot_response}\n
    Model: {model_choice}
    """
    )

    ### EVALUATION
    evaluation: List[PredictionResult] = evaluator.evaluate_turn(convo=conversation_history, turn=bot_response)
    evaluator.add_message_evaluation(turn=bot_response, author="bot", evaluation_results=evaluation)
    helpfulness_label = evaluation[2].label

    if helpfulness_label == "Not helpful":
        gr.Warning(f"Bot response '{bot_response}' was flagged not helpful. Generating another bot response...")
        bot_response = query_llm(model_choice, get_prompt_helpfulness_refinement(user_input, bot_response)).strip('"')

    subscore_intent = seed.compute_subscore_intent(evaluator.intent_labels)
    subscore_output = seed.compute_subscore_output_elicitation(conversation_history, evaluator.output_labels)
    subscore_helpfulness = seed.compute_subscore_helpfulness(evaluator.helpfulness_labels)

    seed_total = seed.compute_total_seed(subscore_intent, subscore_output, subscore_helpfulness)
    logger.info(f"Seed total: {seed_total}")
    conversation_history.pop()

    return (
        bot_response,
        format_highlight_evaluation_results(evaluation),
        format_seed_results(seed_total, subscore_intent, subscore_output, subscore_helpfulness),
    )


def build_chat_ui():
    with gr.Blocks(theme="ocean") as demo:
        default_context, default_requirements = load_config()
        conv_id = datetime.now().strftime("%Y%m%d%H%M%S")
        gr.HTML(f"<style>{get_css(__file__)}</style>")

        evaluation_box, seed_box = create_evaluation_module()

        with gr.Row():
            gr.Markdown("# Bianca Ciobanica - Unified Evaluator and SEED Demo")

        with gr.Row():
            ### LEFT COLUMN ### Prompt Configuration
            with gr.Column():
                gr.Markdown("### Prefix Configuration")

                with gr.Column(elem_id="prompt-column"):  # Main layout with 3 columns
                    prompt_preview = create_prompt_preview_module(default_context, default_requirements)
                    with gr.Column(scale=1, min_width=250):
                        context_input, requirements_input, notification_box = create_prompt_parameters_module(
                            default_context, default_requirements, prompt_preview
                        )
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

                save_conversation_button.click(
                    fn=lambda: save_conversation_with_evaluation(evaluator.evaluated_conversation, conv_id),
                    outputs=[notification_box],
                )

            ### RIGHT COLUMN ### Evaluation
            with gr.Column(scale=1, min_width=250):
                gr.Markdown("### Evaluation Results")
                evaluation_box.render()

                gr.Markdown("### SEED Results")
                seed_box.render()

        return demo
