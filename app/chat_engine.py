import re
import textwrap
from datetime import datetime
from typing import List

import gradio as gr

from app.components.about_info_ui import add_info_about_app
from app.components.chat_ui import create_chat_module
from app.components.evaluation_ui import create_evaluation_module
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
    task = f"""\nTask: generate a new bot response that meets the previous requirements. 
    Output only the bot response and nothing else. Do not print explanations.\n
            Bot response: {bot_response}
"""
    refinement_instructions += f"The previous content was: {history}\n"

    if ACTION_HELPFUL in labels:
        refinement_instructions += textwrap.dedent(
            f"""
        A helpful response is considered as helpful if:
        - it conveys one idea at a time, 
        - uses simple and easy to understand language
        - is not ambiguous (direct language)\n
    """
        )

    if ACTION_OUTPUT in labels:
        refinement_instructions += f"The bot response is not eliciting output. Keep the bot response and append to it a question or an imperative statement to the end of the response that would be a great continuity to the current content.\n"

    if ACTION_INTENT in labels:
        refinement_instructions += f"The bot response lacks describing someone or introducing someone as intent."

    refinement_instructions += task
    return refinement_instructions


def end_conversation_prompt(history):
    return f"""
    Given this conversation in a role-play game: 
    \nHistory: \n
        {history}\n
    
    Give a response as that ends the conversation smoothly with the player as the role Bot. Output only the response and nothing else. Do not print explanations."""


def generate_bot_response(history, model_choice, context, requirements):
    structured_prompt = create_prompt_template(context, requirements, history)

    bot_response = query_llm(model_choice, structured_prompt).strip('"')  # we need user input for generating a response
    bot_response = re.sub(r"\*(.*?)\*", "", bot_response)  # avoid inline actions

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


def get_new_evaluation(history, reason, new_bot_response):
    bad_response = evaluator.evaluated_conversation[-1]
    bad_response.reason_for_bad = ", ".join(reason)
    evaluator.add_bad_evaluation(bad_response)

    evaluator.remove_last_message_evaluation()

    evaluation_of_refined_response = evaluator.evaluate_turn(convo=history, turn=new_bot_response)
    evaluator.add_message_evaluation(
        turn=new_bot_response, author="bot", evaluation_results=evaluation_of_refined_response
    )

    return evaluation_of_refined_response


def chat_and_evaluate(user_input, conversation_history, model_choice, context, requirements):
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

    #### SEED
    logger.info(evaluator.get_conversation_evaluation())
    seed_total = seed.compute_total_seed(
        evaluator.intent_labels, evaluator.output_labels, evaluator.helpfulness_labels, user_replies
    )
    logger.info(f"Seed total: {seed_total}")

    evaluator.add_message_evaluation(turn=bot_response, author="bot", evaluation_results=evaluation, seed_results={"seed":seed_total, "seed_intent": seed.intent_subscore, "seed_output": seed.output_elicitation_subscore, "seed_helpful": seed.helpfulness_subscore})

    actions_for_refinement = get_actions_for_refinement(
        evaluation, seed.intent_subscore, seed.output_elicitation_subscore, seed.helpfulness_subscore
    )

    if len(actions_for_refinement) > 0:
        bot_response = query_llm(
            model_choice, get_prompt_for_refinement(actions_for_refinement, bot_response, formatted_history_string)
        ).strip('"')
        ## update the evaluations with the newly generated bot response (need to do this because seed computes across all turns)
        evaluation = get_new_evaluation(conversation_history, actions_for_refinement, bot_response)
        seed_total = seed.compute_total_seed(
            evaluator.intent_labels, evaluator.output_labels, evaluator.helpfulness_labels, user_replies
        )

    if seed_total >= 0.70 and len(conversation_history) >= 20:
        bot_response = query_llm(model_choice, end_conversation_prompt(formatted_history_string)).strip('"')
        logger.info(f"Ending conversation: {bot_response}")
        evaluator.reset_conversation_evaluation()

    conversation_history.pop()

    return (
        bot_response,
        format_highlight_evaluation_results(evaluation),
        format_seed_results(
            seed_total, seed.intent_subscore, seed.output_elicitation_subscore, seed.helpfulness_subscore
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
                            prompt_preview.render()
                            prompt_preview.change(fn=lambda: trigger_notification("Prompt template updated."))

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
                            save_conversation_with_evaluation(evaluator.get_conversation_evaluation(), conv_id),
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
                eval_history_box = gr.Dataframe(wrap=True, column_widths=[25, 50, 250, 200, 200])
                chatbot.change(fn=evaluator.get_conversation_evaluation_with_bad_responses, outputs=eval_history_box, trigger_mode="always_last")

        with gr.Tab("Prompt Template"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Prompt Configuration")
                    context_input, requirements_input = create_prompt_parameters_module(
                        default_context, default_requirements, prompt_preview
                    )
                    prompt_preview.change(fn=lambda: trigger_notification("Prompt template updated."))
                with gr.Column():
                    gr.Markdown("### Prompt Preview")

                    prompt_preview = create_prompt_preview_module(default_context, default_requirements)
                    prompt_preview.label = "This prompt template is called at each turn for generating a response."
                    prompt_preview.render()

            gr.Markdown("### Bot response refinement config")

        with gr.Tab("About"):
            gr.Markdown(add_info_about_app())

            gr.Markdown("## Unified Model Architecture")
            gr.HTML("""<embed src="app/assets/bert_unified.drawio.pdf" width="100%" height="600px" type="application/pdf">""")
            gr.Markdown("Bianca Ciobanica, 2025")

        return demo
