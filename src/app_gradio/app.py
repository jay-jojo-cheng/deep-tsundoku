import os
from pathlib import Path
from typing import Callable, List

import gradio as gr
from PIL.Image import Image


def main():
    model_inference = ModelInference()
    frontend = make_frontend(model_inference.predict)
    frontend.launch()


def make_frontend(fn: Callable[[Image], str]):
    """Creates a gradio.Interface frontend for an image to text function"""
    # List of example images
    images_dir = os.path.join(get_project_root(), "tests/support")
    example_images = [
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if os.path.splitext(f)[1] in [".jpg", ".jpeg", ".png"]
    ]

    frontend = gr.Interface(
        fn=fn,
        inputs=gr.Image(type="pil", label="Bookshelf"),
        outputs=gr.Textbox(interactive=True, label="Recognized books"),
        title="Bookshelf recognizer 📚",
        examples=example_images,
    )

    return frontend


class ModelInference:
    """
    Runs a machine learning model at inference time. For now, it is simply
    a dummy class that returns the same dummy prediction irrespectively
    of the input.
    """

    def predict(self, image) -> str:
        model_output = ["Book #1", "Book #2", "Book #3"]
        return self._post_process_output(model_output)

    def _post_process_output(self, model_output: List[str]) -> str:
        return "\n".join(model_output)


def get_project_root():
    """Returns the path to the project's root directory: deep-tsundoku"""
    return Path(__file__).parent.parent.parent


if __name__ == "__main__":
    main()
