import gradio as gr


def create_chat_module(chat_and_evaluate, model_choice, context_input, requirements_input, evaluation_results, seed_results):

    # Chatbot
    chatbot = gr.Chatbot(
        type="messages",
        label="NPC",
        avatar_images=("app/assets/avatar_user_icon.jpg", "app/assets/avatar_npc_icon.jpg"),
        bubble_full_width=200,
        editable="user",
    )

    # Chat Interface
    chat_interface = gr.ChatInterface(
        type="messages",
        fn=chat_and_evaluate,
        additional_inputs=[model_choice, context_input, requirements_input],
        additional_outputs=[evaluation_results, seed_results],
        chatbot=chatbot,
    )


    return chatbot, chat_interface
