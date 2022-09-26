import os
from pathlib import Path
from typing import Callable, List

import gradio as gr
from PIL.Image import Image
from gradio import CSVLogger


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

    with gr.Blocks() as frontend:
        #
        # TODO: Add example images
        # TODO: Change layout https://gradio.app/controlling_layout/
        image = gr.Image(type="pil", label="Bookshelf")
        run_button = gr.Button("Find books")
        output = gr.Textbox(label="Recognized books")

        wrong_prediction_button = gr.Button("Flag wrong prediction 🐞")
        user_feedback = gr.Textbox(interactive=True, label="User feedback")

        # Log user feedback
        flag_button = gr.Button("Correct predictions")
        flagging_callback = CSVLogger()
        flag_components = [image, output, user_feedback]
        flagging_callback.setup(flag_components, "user_feedback")
        flag_method = FlagMethod(flagging_callback)
        flag_button.click(
            flag_method,
            inputs=flag_components,
            outputs=[],
            _preprocess=False,
            queue=False,
        )

        # Functionality of buttons
        run_button.click(fn, inputs=image, outputs=output)
        wrong_prediction_button.click(
            lambda model_output: model_output, inputs=output, outputs=user_feedback
        )

    return frontend


class FlagMethod:
    """Copied from gradio's `interface.py` script that mimics the flagging callback"""

    def __init__(self, flagging_callback, flag_option=None):
        self.flagging_callback = flagging_callback
        self.flag_option = flag_option
        self.__name__ = "Flag"

    def __call__(self, *flag_data):
        self.flagging_callback.flag(flag_data, flag_option=self.flag_option)


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
