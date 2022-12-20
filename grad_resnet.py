import os
import sys
import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

input_path = sys.argv[1]
img_name = os.path.join('./input', input_path)

pil_img = PIL.Image.open(img_name)
pil_img

torch_img = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])(pil_img).to(device)
normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]


resnet18 = models.resnet18(pretrained=True)
resnet50 = models.resnet50(pretrained=True)
resnet101 = models.resnet101(pretrained=True)
resnet152 = models.resnet152(pretrained=True)

configs = [
    dict(model_type='resnet', arch=resnet18, layer_name='layer4'),
    dict(model_type='resnet', arch=resnet50, layer_name='layer4'),
    dict(model_type='resnet', arch=resnet101, layer_name='layer4'),
    dict(model_type='resnet', arch=resnet152, layer_name='layer4'),
]

configs_shallow = [
    dict(model_type='resnet', arch=resnet18, layer_name='layer1'),
    dict(model_type='resnet', arch=resnet50, layer_name='layer1'),
    dict(model_type='resnet', arch=resnet101, layer_name='layer1'),
    dict(model_type='resnet', arch=resnet152, layer_name='layer1'),
]

for config in configs:
    config['arch'].to(device).eval()

cams = [
    [cls.from_config(**config) for cls in (GradCAM, GradCAMpp)]
    for config in configs
]

images = []
for gradcam, gradcam_pp in cams:
    mask, _ = gradcam(normed_torch_img)
    heatmap, result = visualize_cam(mask, torch_img)

    mask_pp, _ = gradcam_pp(normed_torch_img)
    heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)
    
    images.extend([torch_img.cpu(), heatmap, heatmap_pp, result, result_pp])
    
grid_image = make_grid(images, nrow=5)
img = transforms.ToPILImage()(grid_image)
img.show()  
