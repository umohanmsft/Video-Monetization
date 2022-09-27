import torch
from torchvision import io
from torchvision import transforms as T
from PIL import Image as im

from semseg import show_models

import gdown
from pathlib import Path
from semseg.models import *
from semseg.datasets import *
import os


outputDirs =  "D://Work//script//infered"
rootaddress = "D://Work//script//filteredImages"
imageWidth = 1280
imageHeight = 720


#   SegFormer
#   Lawin
#   SFNet
#   BiSeNetv1
#   DDRNet
#   FCHarDNet
#   BiSeNetv2

model = eval('SegFormer')(
		backbone='MiT-B3',
		num_classes=150
		)


def getImagesDir():
	imagesDir = os.listdir(rootaddress)
	dirs = []
	for img in imagesDir:
		img = "D://Work//script//filteredImages/"+img
		dirs.append(img)

	return dirs

def save_image(image, path):
	if image.shape[2] == 3: image = image.permute(2, 0, 1)
	# image = im.fromarray(image.numpy())
	print(image.shape)

	io.write_png(image, path)



def show_image(image):
    print(type(image))
    if image.shape[2] != 3: image = image.permute(1, 2, 0)
    image = im.fromarray(image.numpy())
    image.show()
    return image

def downloadModels():
	ckpt = Path('./checkpoints/pretrained/segformer')
	ckpt.mkdir(exist_ok=True, parents=True)
	url = 'https://drive.google.com/uc?id=1-OmW3xRD3WAbJTzktPC-VMOF5WMsN8XT'
	output = './checkpoints/pretrained/segformer/segformer.b3.ade.pth'
	gdown.download(url, output, quiet=False)


def initModels():
	try:
		model.load_state_dict(torch.load('checkpoints/pretrained/segformer/segformer.b3.ade.pth', map_location='cpu'))
	except:
		print("Download a pretrained model's weights from the result table.")
	model.eval()
	show_models()



def inferImages(path):
	image_path = path
	image = io.read_image(image_path)
	print(image.shape)
	# show_image(image)
	#preprocessImages():
	image = T.CenterCrop((imageHeight, imageWidth))(image)
	image = image.float() / 255
	image = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)
	image = image.unsqueeze(0)
	image.shape

	#modelForward():
	with torch.inference_mode():
		seg = model(image)
		seg.shape
	
	seg = seg.softmax(1).argmax(1).to(int)
	seg.unique()
	palette = eval('ADE20K').PALETTE
	seg_map = palette[seg].squeeze().to(torch.uint8)
	# print(seg_map)
	# show_image(seg_map)
	# print(seg_map.shape)
	return seg_map


initModels()
images_dir = getImagesDir()

i = 0
for item in images_dir:
	imageToSave = inferImages(item)
	outputDir = item.replace("filteredImages", "infered")
	save_image(imageToSave, outputDir)
	print("*****************")
	i+=1
	print("Infered Image "+str(i))

# Analyse Colour
# -Load Image and get RGB Value
# -Load Image and Get the dimension where to upload
# -Calculate duration of Add 
# -Convert To Video needed ?

