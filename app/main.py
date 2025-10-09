#%%
import gradio as gr
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np

# Example top prediction
cyano_bac_ex = {'class': 'mycrocystis', 'prob': 0.78}

# Dummy exemplar database
ds_dir = "./../dataset/LaboCDD31_CyaonBacteries/"
exemplar_db = {
    "mycrocystis": [ds_dir+"116011 Microcystis.bmp",ds_dir+"116011 Microcystis.bmp",ds_dir+"116011 Microcystis.bmp",ds_dir+"116011 Microcystis.bmp",ds_dir+"116011 Microcystis.bmp"],
    "aphanizomenon": [ds_dir+"116014 Aphanizomenon.bmp",ds_dir+"116014 Aphanizomenon.bmp",ds_dir+"116014 Aphanizomenon.bmp",ds_dir+"116014 Aphanizomenon.bmp",ds_dir+"116014 Aphanizomenon.bmp"],
    "woronichinia": [ds_dir+"116014 Woronichinia.bmp",ds_dir+"116014 Woronichinia.bmp",ds_dir+"116014 Woronichinia.bmp",ds_dir+"116014 Woronichinia.bmp",ds_dir+"116014 Woronichinia.bmp"],
}

#%%
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
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=40)
    except:
        font = ImageFont.load_default()
    draw.text((10, 10), f"{cyano_bac_ex['class']}: {cyano_bac_ex['prob']:.2f}", fill="red", font=font)
    variants.append(noisy_img)

    bw_img = ImageOps.grayscale(img).convert("RGB")
    draw = ImageDraw.Draw(bw_img)
    draw.text((10, 10), f"{cyano_bac_ex['class']}: {cyano_bac_ex['prob']:.2f}", fill="red", font=font)
    variants.append(bw_img)

    return variants

def get_top3_predictions(image_index):
    top3 = [("mycrocystis", 0.78), ("aphanizomenon", 0.15), ("woronichinia", 0.07)]
    return top3

# When user clicks a variant image
def on_image_click(image_index):
    top3 = get_top3_predictions(image_index)
    classes = [cls+ ": " +str(prob) for cls, prob in top3]
    radio_update = gr.update(choices=classes, value=None, visible=True)

    exemplars = []
    for cls,_ in top3:
        for i in range(5):
            exemplars.append(Image.open(exemplar_db[cls][i]).convert("RGB"))
    return radio_update, exemplars
    # print(f"User labeled image {image_index} as {chosen_class}")
    # return "Saved!"

def submit_user_choice(image_index, chosen_class):
    if chosen_class is None:
        return "No class selected!"
    
    # Here you can save the data to a CSV, JSON, or database
    # For example, just print for now
    print(f"User labeled image {image_index} as {chosen_class}")
    
    return "Saved!"

def show_temporary_message(image_index, chosen_class):
    if chosen_class is None:
        return gr.update(value="", visible=False)

    msg = f"Saved correction: {chosen_class}"
    
    # Return an HTML string that fades out
    fade_html = f"""
    <div id="fade-message" style="
        animation: fadeOut 2.5s forwards;
        color: white;
        background-color: #4CAF50;
        padding: 15px 30px;
        border-radius: 10px;
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 9999;
        font-weight: bold;
        font-size: 18px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    ">
        {msg}
    </div>
    <style>
        @keyframes fadeOut {{
            0% {{ opacity: 1; }}
            70% {{ opacity: 1; }}
            100% {{ opacity: 0; visibility: hidden; }}
        }}
    </style>
    """

    return gr.update(value=fade_html, visible=True)

with gr.Blocks(title="Cyanobacteria Visual Tool", css_paths="style.css") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 🧫 Cyanobacteria Visual Tool")
            gr.Markdown("Upload microscopy images of cyanobacteria to visualize or annotate.")

        # feedback_text = gr.Textbox(label="Status")
        feedback_html = gr.HTML(value="", visible=False)

    with gr.Row():
        with gr.Column(scale=1):
            image_gallery = gr.Gallery(
                label="Generated Variants",
                show_label=True,
                height=400,            
                columns=1,            
                object_fit="contain"  
            )
            top3_radio = gr.Radio(label="Select Correct Class", choices=[])
            submit_btn = gr.Button("Submit Correction")
            # submit_btn.click(
            #     fn=submit_user_choice,
            #     inputs=[image_gallery, top3_radio],  # gallery index + selected class
            #     outputs=feedback_text                # display a status message
            # )
            submit_btn.click(
                fn=show_temporary_message,
                inputs=[image_gallery, top3_radio],
                outputs=feedback_html
            )

        with gr.Column(scale=2):
            image_component = gr.Image(
                label="Upload or Drop an Image",
                type="filepath",
                height=400,
                interactive=True
            )
            process_btn = gr.Button("Process Image", elem_classes="my-btn")

            process_btn.click(fn=show_images, inputs=image_component, outputs=image_component)
            process_btn.click(fn=generate_variants, inputs=image_component, outputs=image_gallery)

    with gr.Row():
        exemplar_gallery = gr.Gallery(label="Exemplar Images", show_label=True, height=300, columns=5)
        image_gallery.select(on_image_click, inputs=image_gallery, outputs=[top3_radio, exemplar_gallery])

if __name__ == "__main__":
    demo.launch()
