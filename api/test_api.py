#%%
import requests
from PIL import Image
import io 
import base64


API_URL = "http://localhost:5075/predict"
IMAGE_PATH = r"../data/VOC/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000089.jpg"
 


# Convert image to bytes
# image = Image.open(IMAGE_PATH)
# img_binary = io.BytesIO()
# image.save(img_binary, format="PNG")

# response = requests.post("http://0.0.0.0:5075/predict",
#                          files={"file": img_binary.getvalue()},
#                          data={"iou_threshold": iou_threshold,
#                                "score_threshold": score_threshold}
#                         )

buff = io.BytesIO()
pil_image = Image.open(IMAGE_PATH)
pil_image.save(buff, format="JPEG")
img_base64_string = base64.b64encode(buff.getvalue()).decode('utf-8')
image_raw_bytes = base64.b64decode(img_base64_string)
iou_threshold = 0.49
score_threshold = 0.79

# Send request to the API
response = requests.post("http://0.0.0.0:5075/predict",
                         files={"file": ("image.png", image_raw_bytes)},
                         data={"iou_threshold": iou_threshold,
                               "score_threshold": score_threshold}
                        )


print("Status:", response.status_code)
print("Response:", response.json())
# %%
