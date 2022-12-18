import os
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

img_name = './input/crop_100.png'

pil_img = PIL.Image.open(img_name)
pil_img

torch_img = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])(pil_img).to(device)
normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]


alexnet = models.alexnet(pretrained=True)
vgg = models.vgg16(pretrained=True)
resnet = models.resnet101(pretrained=True)
densenet = models.densenet161(pretrained=True)
squeezenet = models.squeezenet1_1(pretrained=True)

configs = [
    dict(model_type='alexnet', arch=alexnet, layer_name='features_11'),
    dict(model_type='vgg', arch=vgg, layer_name='features_29'),
    dict(model_type='resnet', arch=resnet, layer_name='layer4'),
    dict(model_type='densenet', arch=densenet, layer_name='features_norm5'),
    dict(model_type='squeezenet', arch=squeezenet, layer_name='features_12_expand3x3_activation')
]

configs_shallow = [
    dict(model_type='alexnet', arch=alexnet, layer_name='features_0'),
    dict(model_type='vgg', arch=vgg, layer_name='features_0'),
    dict(model_type='resnet', arch=resnet, layer_name='layer1'),
    dict(model_type='densenet', arch=densenet, layer_name='features_norm0'),
    dict(model_type='squeezenet', arch=squeezenet, layer_name='features_3_expand3x3_activation')
]

for config in configs_shallow:
    config['arch'].to(device).eval()

cams = [
    [cls.from_config(**config) for cls in (GradCAM, GradCAMpp)]
    for config in configs_shallow
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
