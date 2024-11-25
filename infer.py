import argparse
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import os
import cv2
import matplotlib.pyplot as plt

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.model = smp.UnetPlusPlus(
        encoder_name="resnet50",        
        encoder_weights="imagenet",      
        in_channels=3, 
        classes= 3,
    )

    def forward(self, x):
        return self.model(x)

def load_model(model_path, DEVICE):
    model = UNet()
    model.load_state_dict(torch.load(model_path))
    model = model.to(DEVICE)  
    model.eval()
    return model

def preprocess_image(image_path, target_size=(256, 256)):
    image = Image.open(image_path).convert("RGB")  
    preprocess = transforms.Compose([ 
        transforms.Resize(target_size), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = preprocess(image).unsqueeze(0) 
    return image

def postprocess_output(output, threshold=0.5):
    output = output.squeeze(0).cpu().detach().numpy()  
    output = (output > threshold).astype(np.uint8)  
    return output

# Map segmentation output to colors (3 classes: 0: background, 1: neopolyp, 2: non-neopolyp)
def map_output_to_color(output):
    output_color = np.zeros((output.shape[1], output.shape[2], 3), dtype=np.uint8)  # empty color image
    output_color[output == 1] = [255, 0, 0]  # Neopolyp - Red
    output_color[output == 2] = [0, 255, 0]  # Non-neopolyp - Green
    return output_color

def infer(model, image_path, DEVICE, output_path="output_segmented_image.png"):
    input_image = preprocess_image(image_path)  
    input_image = input_image.to(DEVICE) 

    with torch.no_grad():
        output = model(input_image) 

    output_mask = postprocess_output(output)  
    output_color = map_output_to_color(output_mask) 

    plt.imshow(output_color)  
    plt.axis('off') 
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)  

    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for image segmentation.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image")
    parser.add_argument('--checkpoint', type=str, default="./checkpoints/model.pth", help="Path to the model checkpoint")
    args = parser.parse_args()  
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, DEVICE)
    infer(model, args.image_path, DEVICE)