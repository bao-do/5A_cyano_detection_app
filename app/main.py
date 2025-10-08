import gradio as gr
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np


cyano_bac_ex = {'class': 'mycrocystis', 'prob': 0.78}

def show_images(image_path):
    if image_path is None:
        return None
    image = Image.open(image_path)
    return image


def generate_variants(image_path):
    if image_path is None:
        return []
    variants = []
    img = Image.open(image_path).convert("RGB")

    np_img = np.array(img) / 255.0
    noise = np.random.normal(0, 0.1, np_img.shape)
    noisy_img = np.clip(np_img + noise, 0, 1) * 255
    noisy_img = Image.fromarray(noisy_img.astype(np.uint8))

    draw = ImageDraw.Draw(noisy_img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=60)
    draw.text((0, 0), f"{cyano_bac_ex['class']}: {cyano_bac_ex['prob']}", fill="red", font=font)
    variants.append(noisy_img)

    bw_img = ImageOps.grayscale(img)
    bw_img = bw_img.convert("RGB")  
    draw = ImageDraw.Draw(bw_img)
    draw.text((0, 0), f"{cyano_bac_ex['class']}: {cyano_bac_ex['prob']}", fill="red", font=font)
    variants.append(bw_img)
    
    return variants


with gr.Blocks(title="Cyanobacteria Visual Tool", css_paths="style.css") as demo:
    gr.Markdown("## 🧫 Cyanobacteria Visual Tool")
    gr.Markdown("Upload microscopy images of cyanobacteria to visualize or annotate.")

    with gr.Row():
        with gr.Column(scale=1):
            image_gallery = gr.Gallery(
                label="Generated Variants",
                show_label=True,
                height=400,            
                columns=1,            
                object_fit="contain"  
            )

        with gr.Column(scale=2):
            image_component = gr.Image(
                label="Upload or Drop an Image",
                type="filepath",
                height=400,
                interactive=True
            )
            process_btn = gr.Button("Process Image", elem_classes="my-btn")

            process_btn.click(
                fn=show_images,
                inputs=image_component,
                outputs=image_component
            )

            process_btn.click(
                fn=generate_variants,
                inputs=image_component,
                outputs=image_gallery
            )

if __name__ == "__main__":
    demo.launch()
