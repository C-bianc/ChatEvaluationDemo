import gradio as gr


def create_evaluation_module():
    evaluation_box = gr.Highlightedtext(
        label="Response Quality (Unified model as evaluator)",
        interactive=False,
        color_map={"argmax": "green", "next_highest": "yellow", "third_highest": "red"},
        show_legend=False,
        show_inline_category=False,
        render=False,
        min_width=400,
        elem_id="evaluation-box",
    )

    seed_scores = gr.Highlightedtext(
        label="Seed Score (normalized)", interactive=False, render=False, min_width=400, elem_id="evaluation-box",
        color_map={"satisfying": "green", "enough": "yellow", "not enough": "red", "subscore": "lightgrey"},

    )

    return evaluation_box, seed_scores
