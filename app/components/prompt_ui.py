import gradio as gr

from app.utils.conversation import save_context_config, save_requirements_config, create_prompt_template


def create_prompt_parameters_module(default_context, default_requirements, prompt_preview_component):

    notification_box = gr.Textbox(label="Status", visible=True, interactive=False)

    # Context input
    context_input = gr.Textbox(label="Conversation Context", value=default_context, lines=3)

    # Requirements input
    requirements_input = gr.Textbox(label="Constraints", value=default_requirements, lines=5)

    # Add change event handlers AFTER all components are defined
    context_input.change(
        fn=save_context_config,
        inputs=[context_input],
        outputs=[notification_box, prompt_preview_component],  # Include prompt_preview here
    )

    requirements_input.change(
        fn=save_requirements_config,
        inputs=[requirements_input],
        outputs=[notification_box, prompt_preview_component],  # Include prompt_preview here
    )
    return context_input, requirements_input, notification_box

def create_prompt_preview_module(default_context, default_requirements):
    default_prompt = create_prompt_template(default_context, default_requirements, None)

    # Prompt preview
    prompt_preview = gr.Textbox(
        label="Current Prompt",
        lines=8,
        interactive=False,
        render=False,
        value=default_prompt
    )
    return prompt_preview
