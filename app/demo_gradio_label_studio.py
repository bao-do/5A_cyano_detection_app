# from label_studio_sdk import LabelStudio

# # Connect to your local Label Studio instance
# ls = LabelStudio(
#     base_url="http://localhost:8080",
#     api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA2ODY4NjAzMiwiaWF0IjoxNzYxNDg2MDMyLCJqdGkiOiJiYzUzY2JlZDQ2OTA0MWVlOTgwYWNhZjI1ZjQzNzAyZSIsInVzZXJfaWQiOjF9.VMyniQdtCddsP0qFRGYEkFGa3HOKoqzrAoF1lbhmnrk"
# )

# # Project ID where your labeling interface is already defined
# PROJECT_ID = 1

# # Create a text classification task
# task = ls.tasks.create(
#     project=1,
#     data={
#         "image": "/data/local-files?d=VOC/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000002.jpg"
#     }
# )
# print(task)

import gradio as gr
import os
from label_studio_sdk import LabelStudio
from PIL import Image


abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ds_dir = os.path.join(abs_path, "data/raw_images")


def show_and_save_image(image_path):
    if image_path is None:
        return None
    # Load and preprocess
    image = Image.open(image_path)
    
    image.save(os.path.join(ds_dir,os.path.basename(image_path)))
    return image




with gr.Blocks() as demo:
    image_component = gr.Image(
        label="Upload or Drop an Image",
        type="filepath",
        height=400,
        interactive=True
    )
    process_btn = gr.Button("Process Image", elem_classes="my-btn")
    
    process_btn.click(fn=show_and_save_image, inputs=image_component, outputs=image_component)

if __name__ == "__main__":
    demo.launch()