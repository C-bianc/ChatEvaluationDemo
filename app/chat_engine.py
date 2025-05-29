import gradio as gr
from ollama import generate

from constants import LLMS
from evaluator import ConversationEvaluator
from utils import (
    create_prompt_template,
    load_config,
    save_context_config,
    save_requirements_config,
)


# TODO: add user message to the interface immediately after being sent
# TODO: clean user input after submitting + add input button
# TODO: clean code? check why slow,
#  and also keep evaluation history? (it deletes every time after new message

# TODO: ultimate task, add an llm evaluator (if not enough output after 3 turns, then DSPY, for prompt modif
# TODO: add prompt modif interface (prompt context, restart a conv with the prompt


def query_llm(model_choice, prompt):
    response = generate(model=model_choice, prompt=prompt)
    return response.response


# Full chat-and-evaluate pipeline
def chat_and_evaluate(user_input, conversation_history, model_choice, context, requirements):
    if not conversation_history:
        user_prompt = {"role": "user", "content": user_input}
        conversation_history.append(user_prompt)

    structured_prompt = create_prompt_template(context, requirements, conversation_history)

    bot_response = query_llm(model_choice, structured_prompt)  # conv settings + history + constraints

    # Evaluate the bot response
    evaluation = evaluator.evaluate_turn(convo=conversation_history, turn=bot_response)

    return bot_response, evaluation


# Gradio UI
# Create your Gradio interface
with gr.Blocks() as demo:
    DEFAULT_CONTEXT, DEFAULT_REQUIREMENTS = load_config()
    evaluator = ConversationEvaluator()

    with gr.Row():
        notification_box = gr.Textbox(label="Status", visible=True, interactive=False)
        prompt_preview = gr.Textbox(
            label="Current Prompt",
            lines=8,
            interactive=False,
            value=create_prompt_template(DEFAULT_CONTEXT, DEFAULT_REQUIREMENTS, None),
        )

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### Prefix Configuration")
            context_input = gr.Textbox(label="Conversation Context", value=DEFAULT_CONTEXT, lines=3)

            context_input.change(
                fn=save_context_config,
                inputs=[context_input],
                outputs=[notification_box, prompt_preview],
            )

            requirements_input = gr.Textbox(
                label="Constraints",
                value=DEFAULT_REQUIREMENTS,
                lines=5,
            )
            requirements_input.change(
                fn=save_requirements_config,
                inputs=[requirements_input],
                outputs=[notification_box, prompt_preview],
            )

        # Chat interface section
        with gr.Column(scale=2, min_width=300):
            with gr.Column():
                gr.Markdown("### Bot Evaluation Results")
                evaluation_box = gr.Textbox(
                    label="Evaluation",
                    interactive=False,
                    lines=5,
                    info="Unified model as evaluator.",
                )
            # Model selection
            model_dropdown = gr.Dropdown(choices=LLMS, value=LLMS[0], label="Select a Model")
            gr.Markdown("### Chat Interface")

            # Chatbot
            chatbot = gr.Chatbot(
                type="messages",
                label="NPC",
                avatar_images=("user.png", "npc.png"),
            )

            chat_interface = gr.ChatInterface(
                chat_and_evaluate,
                additional_inputs=[model_dropdown, context_input, requirements_input],
                additional_outputs=[evaluation_box],
                chatbot=chatbot,
                theme="ocean",
                type="messages",
            )

demo.launch()
