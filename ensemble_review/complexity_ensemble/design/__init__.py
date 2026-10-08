"""Visual design for the Gradio application."""

from pathlib import Path


def launch_style() -> dict:
    """Return Gradio 6 launch options for the light, flat theme."""
    import gradio as gr

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.green,
        secondary_hue=gr.themes.colors.gray,
        neutral_hue=gr.themes.colors.gray,
        radius_size=gr.themes.sizes.radius_md,
        spacing_size=gr.themes.sizes.spacing_md,
    ).set(
        body_background_fill="#fafafa",
        background_fill_primary="#ffffff",
        background_fill_secondary="#ffffff",
        panel_background_fill="#ffffff",
        block_background_fill="#ffffff",
        input_background_fill="#ffffff",
        block_border_color="#e5e7eb",
        panel_border_color="#e5e7eb",
        color_accent="#4C820D",
        color_accent_soft="#f3f8eb",
        border_color_accent="#4C820D",
        checkbox_background_color_selected="#4C820D",
        checkbox_border_color_selected="#4C820D",
        checkbox_border_color_focus="#4C820D",
        checkbox_label_background_fill_selected="#f3f8eb",
        checkbox_label_border_color_selected="#4C820D",
        checkbox_label_text_color_selected="#365f0a",
        input_border_color_focus="#4C820D",
        button_primary_background_fill="#eaf7ed",
        button_primary_background_fill_hover="#d9f0df",
        button_primary_border_color="#cee9d5",
        button_primary_text_color="#166534",
        button_secondary_background_fill="#ffffff",
        button_secondary_background_fill_hover="#f9fafb",
        loader_color="#2563eb",
        shadow_drop="none",
        shadow_drop_lg="none",
    )
    default_light = """() => {
        const url = new URL(window.location.href);
        if (!url.searchParams.has("__theme")) {
            url.searchParams.set("__theme", "light");
            window.location.replace(url.toString());
        }
    }"""
    return {
        "theme": theme,
        "js": default_light,
        "css_paths": Path(__file__).with_name("loading.css"),
    }
