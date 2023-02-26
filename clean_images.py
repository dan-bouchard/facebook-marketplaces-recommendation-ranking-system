import os
from tqdm import tqdm
from PIL import Image
from zipfile import ZipFile
from torchvision.transforms import ToTensor
import torch

def resize_image(im, final_size=512):
    im = im.resize((final_size, final_size))
    if im.mode != 'RGB':
        im = im.convert(mode='RGB')
    return im

def generate_resized_images(final_size=512):
    with ZipFile('./images_fb.zip') as myzip:
        img_filenames = myzip.namelist()

    img_filenames.remove('images/')
    img_filenames.remove('images/Links.csv')

    if not os.path.exists(f'./cleaned_images_{final_size}'):
        os.makedirs(f'./cleaned_images_{final_size}')

    with ZipFile('./images_fb.zip') as myzip:
        for img_filename in tqdm(img_filenames):
            image = myzip.open(img_filename)
            im = resize_image(Image.open(image), final_size)
            im.save(f'./cleaned_images_{final_size}/{img_filename[7:]}')

def image_processor(img):
    transform = ToTensor()
    img_tensor = transform(img)
    img_tensor_reshape = torch.unsqueeze(img_tensor, dim=0)
    return img_tensor_reshape