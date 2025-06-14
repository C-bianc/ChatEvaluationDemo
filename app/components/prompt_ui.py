import gradio as gr

from app.utils.conversation import save_context_config, save_requirements_config, create_prompt_template


def create_prompt_parameters_module(default_context, default_requirements, prompt_preview_component):

    # Context input
    context_input = gr.Textbox(label="Conversation Context", value=default_context, lines=20)

    # Requirements input
    requirements_input = gr.Textbox(label="Constraints", value=default_requirements, lines=6)

    # Add change event handlers AFTER all components are defined
    context_input.change(
        fn=save_context_config,
        inputs=[context_input],
        outputs=[prompt_preview_component],  # Include prompt_preview here
    )

    requirements_input.change(
        fn=save_requirements_config,
        inputs=[requirements_input],
        outputs=[prompt_preview_component],  # Include prompt_preview here
    )
    return context_input, requirements_input

def create_prompt_preview_module(default_context, default_requirements):
    default_prompt = create_prompt_template(default_context, default_requirements, None)

    # Prompt preview
    prompt_preview = gr.Textbox(
        label="Current Prompt",
        lines=25,
        interactive=False,
        render=False,
        value=default_prompt
    )
    return prompt_preview

def create_prompt_for_refinement_preview():
    return gr.Textbox(
        label="Prompt for refinement",
        lines=16,
        interactive=False,
        value=""
    )

