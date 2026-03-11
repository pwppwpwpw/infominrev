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
import net
import torchvision.utils as utils
from torch.autograd import Variable


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train_transform(img_size=224):

    transform_list = []  
    transform_list.append(transforms.Resize(size = [img_size, img_size]))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean= [0.485, 0.456, 0.406] ,std= [0.229, 0.224, 0.225]))
     
    return transforms.Compose(transform_list)

def content_transform(h,w):
    h=h//8*8
    w=w//8*8
    
    transform_list = []   
    transform_list.append(transforms.Resize((h,w)))
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
# Basic options  "./testData/random-8331/content-50"  ./testData/content-6
parser.add_argument('--content', type=str, default ="./testData/content-6",
                    help='File path to the content image')
parser.add_argument('--style', type=str, default ="./testData/style-6",
                    help='File path to the style image, or multiple style \
  #                  images separated by commas if you want to do style \
   #                 interpolation or spatial control')
# glow(1).pth 在class Flow(nn.Module)中的use_coupling应该要设置为False
parser.add_argument('--resume', default='glow_ori.pth', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--save_dir', default='./experiments+safin(newxukai)',help='Directory to save the model')
parser.add_argument('--steps', type=str, default = 1)
parser.add_argument('--vgg', type=str, default = 'model/vgg_normalised.pth')
parser.add_argument('--start_iter', type=float, default=1)
parser.add_argument('--max_iter', type=int, default=50)
# Additional options
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default = '.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default ='xukaitest1',
                    help='Directory to save the output image(s)')
parser.add_argument('--n_flow', default=10, type=int, help='number of flows in each block')# 32
parser.add_argument('--n_block', default=2, type=int, help='number of blocks')# 4
parser.add_argument('--no_lu', action='store_true', help='use plain convolution instead of LU decomposed version')
parser.add_argument('--affine', default=False, type=bool, help='use affine coupling instead of additive')
parser.add_argument('--operator', type=str, default='adain',
                    help='style feature transfer operator')
parser.add_argument('--out_dir', type=str, default="output")
parser.add_argument('--label_mapping', type=str, default='models/segmentation/ade20k_semantic_rel.npy')
parser.add_argument('--palette', type=str, default='models/segmentation/ade20k_palette.npy')
parser.add_argument('--min_ratio', type=float, default=0.02)

# Advanced options

args = parser.parse_args()
args.resume = os.path.join(args.save_dir, args.resume)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# os.environ['CUDA_LAUNCH_BLOCKING'] = '7'

save_dir = './salient transfer/good-6-content-size-right-w&h_glow(1)'
os.makedirs(save_dir, exist_ok=True)

if not os.path.exists(args.output):
    os.mkdir(args.output)
vgg=net.vgg
network = net.Net(vgg,8, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)
# network = net.Net(vgg,3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)
print(args)
# -----------------------resume training------------------------
if os.path.isfile(args.resume):
    print("--------loading checkpoint----------")
    print('checkpoint path: ', args.resume)
    checkpoint = torch.load(args.resume)
    args.start_iter = checkpoint['iter']
    network.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}'".format(args.resume))
else:
    print("--------no checkpoint found---------")
network = network.to(device)

network.eval()


content_paths=Path(args.content)
#content_paths = [f for f in content_paths.glob('*')]
style_paths=Path(args.style)
#style_paths = [f for f in style_paths.glob('*')]

lc1 = 0
ls1 = 0
j=0
ls_list = []
averge_region = []
#content = content_tf(Image.open(args.content))
#style = style_tf(Image.open(args.style))

#style = style.to(device).unsqueeze(0)
#content = content.to(device).unsqueeze(0)

content_name_list = os.listdir(content_paths)
style_name_list = os.listdir(style_paths)

numbers = len(content_name_list) if len(content_name_list) < len(style_name_list) else len(style_name_list)

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
out_dir = args.out_dir

 #  segmentation
# You can use any 'ade20k' segmentation model
# -----------------------------
# An example of using SegFormer
print("Building Salinet Model")
import sys
sys.path.append('..')
sys.path.append('../M3Net/')
from M3Net import M3Net
# checkpoint = r'D:\Code\VScode\Xiangxiu-region\M3Net\savepth\M3Net-R.pth'
checkpoint = r'D:\Code\VScode\Xiangxiu-region\M3Net\savepth\swim-384-scratch-b8M3Net-S.pth'
# model = M3Net(embed_dim=384,dim=64,img_size=224,method='M3Net-R').to(device)
model = M3Net(embed_dim=512,dim=64,img_size=384,method='M3Net-S').to(device)
model.load_state_dict(torch.load(checkpoint))
model.eval()
# seg_model = init_segmentor(config, checkpoint, device=device)
train_tf = train_transform(img_size=384)
content_mask_save_dir = './salient transfer/content-4-mask-right-w&h'
os.makedirs(content_mask_save_dir, exist_ok=True)

style_mask_save_dir = './salient transfer/style-4-mask-right-w&h'
os.makedirs(style_mask_save_dir, exist_ok=True)

to_PIL = transforms.ToPILImage()

content_region = []
style_region = []

for i in range(0,numbers):
    content_path=os.path.join(content_paths, content_name_list[i])
    style_path=os.path.join(style_paths, style_name_list[i])  

    with torch.no_grad():

        content_rgb =Image.open(content_path).convert("RGB")
        style_rgb = Image.open(style_path).convert("RGB")
        # print('style_path:', style_path)

        content_salient = train_tf(content_rgb).to(device).unsqueeze(0)
        style_salient = train_tf(style_rgb).to(device).unsqueeze(0)
        content_salient = Variable(content_salient.cuda())
        style_salient = Variable(style_salient.cuda())

        # Inference
        # content
        content_outputs_saliency = model(content_salient)
        content_mask_1_1 = content_outputs_saliency[-1]
        content_pred = torch.sigmoid(content_mask_1_1)

        # print('content_pred min: {}, content_pred max: {}, content_pred mid: {}'.format(torch.min(content_pred), torch.max(content_pred)
        #                                                                                 ,content_pred.quantile(q=0.5).item()))
        content_pred[content_pred < 70/255 ] = 0
        content_pred[content_pred > 70/255 ] = 1
        # content_w, content_h = content_rgb.size
        content_h, content_w = content_rgb.size
        
        content_salient_transform = transforms.Compose([
            transforms.Resize((content_w, content_h), interpolation=Image.NEAREST),
        ])

        content_pred = content_pred.squeeze(0)
        content_pred = content_salient_transform(content_pred)
        # content_mask = to_PIL(content_pred)
        # content_mask.save(os.path.join(content_mask_save_dir, content_name_list[i]))
        
        # style
        style_outputs_saliency = model(style_salient)
        style_mask_1_1 = style_outputs_saliency[-1]
        style_pred = torch.sigmoid(style_mask_1_1)

        style_pred[style_pred < 70/255 ] = 0
        style_pred[style_pred > 70/255 ] = 1
        # style_w, style_h = style_rgb.size
        style_h, style_w = style_rgb.size
        style_salient_transform = transforms.Compose([
            transforms.Resize((style_w, style_h), interpolation=Image.NEAREST),
        ])

        style_pred = style_pred.squeeze(0)
        style_pred = style_salient_transform(style_pred)
        # style_mask = to_PIL(style_pred)
        # style_mask.save(os.path.join(style_mask_save_dir, style_name_list[i]))
        # print('style_pred min: {}, style_pred max: {}, style_pred mid: {}'.format(torch.min(style_pred), torch.max(style_pred)
        #                                                                           ,style_pred.quantile(q=0.5).item()))
        # continue
        # -----------------------------


        # Post-processing segmentation results
        content_seg = np.asarray(content_pred.cpu()).astype(np.uint8)
        style_seg = np.asarray(style_pred.cpu()).astype(np.uint8)

        current_region_number = len(np.unique(content_seg))
        current_style_number = len(np.unique(style_seg))
        content_region.append(current_region_number)
        style_region.append(current_style_number)
        # print('**********************************************************')
        # print('content_seg', np.unique(content_seg))
        # print('style_seg', np.unique(style_seg))
        # print('**********************************************************')

        # Save the class label of segmentation results
        if False:
            if not os.path.exists(os.path.join(out_dir, "segmentation")):
                os.makedirs(os.path.join(out_dir, "segmentation"))
            Image.fromarray(content_seg).save(os.path.join(out_dir, "segmentation", 'content_seg_label.png'))
            Image.fromarray(style_seg).save(os.path.join(out_dir, "segmentation", 'style_seg_label.png'))

        # Save the visualization of segmentation results
        if False:
            palette = np.load(args.palette)
            content_seg_color = np.zeros((content_seg.shape[0], content_seg.shape[1], 3), dtype=np.uint8)
            for label, color in enumerate(palette):
                content_seg_color[content_seg == label, :] = color  # RGB
            Image.fromarray(content_seg_color).save(os.path.join(out_dir, "segmentation", 'content_seg_color.png'))

            style_seg_color = np.zeros((style_seg.shape[0], style_seg.shape[1], 3), dtype=np.uint8)
            for label, color in enumerate(palette):
                style_seg_color[style_seg == label, :] = color  # RGB
            Image.fromarray(style_seg_color).save(os.path.join(out_dir, "segmentation", 'style_seg_color.png'))

        # content = transforms.ToTensor()(content).unsqueeze(0).to(device)
        # style = transforms.ToTensor()(style).unsqueeze(0).to(device)

        # content_seg = content_seg[None, ...]    # shape: [B, H, W]
        # style_seg = style_seg[None, ...]

        j += 1
        h, w ,c= np.shape(content_rgb)
        h = h // 8 * 8
        w = w // 8 * 8
        
        # print("**************************")
        # print('h ', h)
        # print('w ', w)
        # print("**************************")
        content_tf = content_transform(h, w)
        style_tf = test_transform((h,w), crop='store_true')
        content = content_tf(content_rgb)
        style = style_tf(style_rgb)
        
        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)
        region_output, ls = network(content, content_seg, style_seg, forward=False, style=style, use_seg=False)
        if torch.isnan(ls):
            print("torch.isnan(ls)")
            continue
        
        ls1 += ls
        ls_list.append(ls)
        print("content image:{},\t style image:{},\t ls:{}\t content region: {}\t style region: {}\t".format(content_name_list[i], style_name_list[i], ls
                                                                                                            ,current_region_number, current_style_number))
        
        # save_name = str(content_name_list[i].split('.')[0]) + "_" + str(style_name_list[i].split('.')[0]) + '.jpg' 
        # utils.save_image(region_output, os.path.join(save_dir, save_name))

print("max ls: {} \t, min ls: {}".format(max(ls_list), min(ls_list)))
print("ls: {}\t, j:{} \t".format(ls1, j))
print()
print("average ls: {}\t average content region: {}\t average style region: {}\t".format(ls1 / j, np.mean(content_region), np.mean(style_region)))
# print("average region number:",sum(averge_region) / len(averge_region))
# 修改了输入图片的大小，原来mask和图片都是224，调整了w，h

"""
glow_salient_global_finetune, swim-384-scratch-b8M3Net-S content/style-50
average ls: 0.6105794906616211   average content region: 2.0     average style region: 1.98


glow_ori, swim-384-scratch-b8M3Net-S content/style-50
average ls: 4.5555338859558105    average content region: 2.0     average style region: 1.98

glow_ori, content/style-50
average ls: 0.4932132959365845   average content region: 2.0     average style region: 1.98
其实?
average ls: 1.5017305612564087   average content region: 2.0     average style region: 1.98

glow_salient_global_scratch_no_supc_salinet_weight_s_2, swim-384-scratch-b8M3Net-S content/style-50
average ls: 1.2718894481658936   average content region: 2.0     average style region: 1.98
"""