import argparse
import os
from pathlib import Path
import glob
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext

#from cv2 import data
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
import net_test as net


def style_transform(h,w):
    k = (h,w)
    size = int(np.max(k))

    transform_list = []    
    transform_list.append(transforms.CenterCrop((h,w)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def content_transform():
    
    transform_list = []   
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

  
def test_transform(size, crop):
    transform_list = []
   
    if size != 0: 
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform
parser = argparse.ArgumentParser()
content_size=512
style_size=512
crop='store_true'
# Basic options
parser.add_argument('--content', type=str, default ="/home/csu_ysy/wct+flow+style1+adapCBAM/input/content",
                    help='File path to the content image')
parser.add_argument('--style', type=str, default ="/home/csu_ysy/wct+flow+style1+adapCBAM/input/style",
                    help='File path to the style image, or multiple style \
  #                  images separated by commas if you want to do style \
   #                 interpolation or spatial control')
parser.add_argument('--resume', default='iter159999_glow.pth', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--save_dir', default='./experiments(s15)',help='Directory to save the model')
parser.add_argument('--steps', type=str, default = 5)
parser.add_argument('--vgg', type=str, default = 'model/vgg_normalised.pth')
parser.add_argument('--embed', type=str, default = 'model/PatchEmbed_iter_160000.pth')
parser.add_argument('--decoder', type=str, default = 'model/decoder_iter_160000.pth')
parser.add_argument('--transform', type=str, default = 'model/transformer_iter_160000.pth')
parser.add_argument('--start_iter', type=float, default=1)
parser.add_argument('--max_iter', type=int, default=50)
# Additional options
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default = '.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default ='ours1',
                    help='Directory to save the output image(s)')
parser.add_argument('--n_flow', default=8, type=int, help='number of flows in each block')# 32
parser.add_argument('--n_block', default=2, type=int, help='number of blocks')# 4
parser.add_argument('--no_lu', action='store_true', help='use plain convolution instead of LU decomposed version')
parser.add_argument('--affine', default=False, type=bool, help='use affine coupling instead of additive')
parser.add_argument('--operator', type=str, default='adain',
                    help='style feature transfer operator')

# Advanced options

args = parser.parse_args()
args.resume = os.path.join(args.save_dir, args.resume)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.exists(args.output):
    os.mkdir(args.output)
vgg=net.vgg
network = net.Net(vgg,8, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)

# -----------------------resume training------------------------
if os.path.isfile(args.resume):
    print("--------loading checkpoint----------")
    checkpoint = torch.load(args.resume)
    args.start_iter = checkpoint['iter']
    network.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}'".format(args.resume))
else:
    print("--------no checkpoint found---------")
network = network.to(device)

network.eval()


content_paths=Path(args.content)
content_paths = [f for f in content_paths.glob('*')]
style_paths=Path(args.style)
style_paths = [f for f in style_paths.glob('*')]

# loss1= lpips.LPIPS(net='alex').to(device)
lossp=0
i=0
#content = content_tf(Image.open(args.content))
#style = style_tf(Image.open(args.style))

#style = style.to(device).unsqueeze(0)
#content = content.to(device).unsqueeze(0)

lc=0
ae=0
ls=0
#loss1= lpips.LPIPS(net='alex').to(device)
lossp=0
i=0
#content = content_tf(Image.open(args.content))
#style = style_tf(Image.open(args.style))

#style = style.to(device).unsqueeze(0)
#content = content.to(device).unsqueeze(0)

for style_path in style_paths:
      
      for content_path in content_paths:
        with torch.no_grad():
         
          i+=1
          content =Image.open(content_path).convert("RGB")
          style = Image.open(style_path).convert("RGB")
          content_tf = test_transform(512, crop='store_true')
          style_tf = test_transform(512, crop='store_true')
          content=content_tf(content)
          style=style_tf(style)
      
          style = style.to(device).unsqueeze(0)
          content = content.to(device).unsqueeze(0)
         
          
          output1,l1,l2,l3= network(content,False,style)
          #aes=aesthetic_encoder(output1)
          #print(aes)
          #ae+=aes
          lc+=l1
          ls+=l2
          
              ##content = content.cpu()
                  #output = torch.cat([style, content, output], 2)
          output_name = '{:s}/{}{}{:s}'.format(
                          args.output, splitext(basename(content_path))[0],
              splitext(basename(style_path))[0],args.save_ext
                      )
          save_image(output1, output_name)
#print("aes",ae/i)
print("lc",lc/i)
print("ls",ls/i)