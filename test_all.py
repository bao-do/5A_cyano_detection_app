#%%
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights


# %%
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, r=4, alpha=1.0):
        super(LoRALinear, self).__init__()
        self.linear = linear_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, linear_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear_layer.out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
    def forward(self, x):
        return self.linear(x) + (self.lora_B @ self.lora_A @ x.transpose(0, 1)).transpose(0, 1) * self.scaling
    
# class LoRAConv2d(nn.Module):
#     def __init__(self, conv_layer, r=4, alpha=1.0):
#         super(LoRAConv2d, self).__init__()
#         self.conv = conv_layer
#         self.r = r
#         self.alpha = alpha
#         self.scaling = alpha / r

#         in_dim = conv_layer.in_channels * conv_layer.kernel_size[0] * conv_layer.kernel_size[1]
#         out_dim = conv_layer.out_channels

#         self.lora_A = nn.Parameter(torch.zeros(r, in_dim,1,1))
#         self.lora_B = nn.Parameter(torch.zeros(out_dim, r,1,1))

#         nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
#         nn.init.zeros_(self.lora_B)


#     def forward(self, x):
#         delta_W  = (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
#         return F.conv2d(x,
#                         self.conv.weight + delta_W,
#                         bias=self.conv.bias,
#                         stride=self.conv.stride,
#                         padding=self.conv.padding,
#                         dilation=self.conv.dilation,
#                         groups=self.conv.groups)

class LoRAConv2d(nn.Module):
    def __init__(self, conv_layer, r=4, alpha=1.0):
        super(LoRAConv2d, self).__init__()
        self.conv = conv_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Use efficient 1x1 convolutions instead of materializing full weight deltas
        # lora_A: 1x1 conv to reduce channels
        self.lora_A = nn.Conv2d(
            conv_layer.in_channels,
            r,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        # lora_B: match original conv's spatial params (stride/padding) so dimensions align
        self.lora_B = nn.Conv2d(
            r,
            conv_layer.out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            bias=False
        )

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

        # Freeze original conv weights
        self.conv.weight.requires_grad = False
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = False

    def forward(self, x):
        # Original convolution output
        out = self.conv(x)
        
        # LoRA adaptation path: reduce channels → apply spatial conv with matching dims
        lora_out = self.lora_A(x)  # [B, r, H, W]
        lora_out = self.lora_B(lora_out)  # [B, out_channels, H', W'] - matches out's shape
        
        # Add scaled LoRA contribution
        return out + lora_out * self.scaling
# %%
model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
for key, value in model.state_dict().items():
    print(f"{key}: {value.shape}")

for name, param in model.named_parameters():
    if not name.startswith("roi_heads"):
        param.requires_grad = False


# %%
num_classes = 21
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
print(f"Number of trainable parameters after replacing the head: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
# %%
r = 8
alpha = 2*r
def replace_module(parent, name, new_module):
    setattr(parent, name, new_module)

resnet = model.backbone.body
layers=("layer3", "layer4")
for layer_name in layers:
    layer = getattr(resnet, layer_name)
    for block in layer:
        for conv_name in ["conv1", "conv2", "conv3"]:
            conv = getattr(block, conv_name)
            lora_conv = LoRAConv2d(conv, r=r, alpha=alpha)
            replace_module(block, conv_name, lora_conv)


rpn_head = model.rpn.head
for block in rpn_head.conv:
    conv = block[0]
    lora_conv = LoRAConv2d(conv, r=r, alpha=alpha)
    block[0] = lora_conv


# box_head = model.roi_heads.box_head
# for idx in range(4):
#     block = box_head[idx]
#     conv = block[0]
#     block[0] = LoRAConv2d(conv, r=r, alpha=alpha)

# linear = box_head[5]
# box_head[5] = LoRALinear(linear, r=r, alpha=alpha)


trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())

print(f"Trainable params: {trainable / 1e6:.2f}M")
print(f"Total params: {total / 1e6:.2f}M")
# %%

dummy_input = torch.randn(1, 3, 224, 224)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
dummy_input = dummy_input.to(device)
model.eval()
with torch.no_grad():
    output = model(dummy_input)
print(output)

# %%
