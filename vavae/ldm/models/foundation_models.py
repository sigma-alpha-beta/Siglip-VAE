import timm
import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModel


def get_mae_encoder():
    """
    Load the MAE pretrained ViT-L encoder from the timm library.
    """
    model = timm.create_model("hf-hub:timm/vit_large_patch16_224.mae", pretrained=True, dynamic_img_size=True)
    model.requires_grad_(False)
    return model

def get_dinov2_encoder():
    """
    Load the DINOv2 pretrained ViT-L encoder from the timm library.
    """
    model = timm.create_model("hf-hub:timm/vit_large_patch14_dinov2.lvd142m", pretrained=True, dynamic_img_size=True)
    model.requires_grad_(False)
    return model

def get_siglip_encoder():
    model_name = "google/siglip2-so400m-patch16-256"
    model = AutoModel.from_pretrained(model_name)
    model.requires_grad_(False)
    return model

def create_foundation_model(
    type,
):
    assert type in ['mae', 'dinov2','siglip'], f"Unsupported foundation model type: {type}"

    if type =='mae':
        return get_mae_encoder(), 1024
    elif type == 'dinov2':
        return get_dinov2_encoder(), 1024
    elif type =='siglip':
        return get_siglip_encoder(), 1152
    
class aux_foundation_model(nn.Module):
    """
    Load the foundation model and forward the input image to get 
    the feature maps.
    """
    def __init__(self, type):
        super().__init__()
        self.model, feature_dim = create_foundation_model(type)
        self.type = type
        self.feature_dim = feature_dim
        if type =='siglip':
            model_name = "google/siglip2-so400m-patch16-256"
            self.processor = AutoProcessor.from_pretrained(model_name)

    def forward_mae(self, x):
        b, c, h, w = x.shape
        return self.model.forward_features(x)[:, 1:].reshape(b, h//16, w//16, -1).permute(0, 3, 1, 2)
    
    def forward_dinov2(self, x):
        b, c, h, w = x.shape
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.model.forward_features(x)[:, 1:].reshape(b, h//16, w//16, -1).permute(0, 3, 1, 2)

    def forward_siglip_image(self, x):
        # x是前处理之后的，用于VF loss和cat
        b, c, h, w = x.shape
        outputs = self.model.vision_model(pixel_values=x)
        return outputs.last_hidden_state.reshape(b, h//16, w//16, -1).permute(0, 3, 1, 2)
    
    def forward_siglip_text(self, texts):
        inputs = self.processor(text=texts, padding="max_length", truncation=True,
                   max_length=64, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        # print('self.model.device:', self.model.device)
        text_vec = self.model.get_text_features(**inputs)
        # text_vec是归一化之前的[b, 1152]，归一化之后算对比损失
        return text_vec
        
    def forward(self, x):
        with torch.no_grad():
            # TODO add siglip model
            if self.type == 'mae':
                return self.forward_mae(x)
            elif self.type == 'dinov2':
                return self.forward_dinov2(x)
            elif self.type =='siglip':
                return self.forward_siglip_image(x)
