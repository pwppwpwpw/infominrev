# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
import models.transformer as transformer
#from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from torchvision.utils import save_image

import net as net
from sampler import InfiniteSamplerWrapper


cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default='/home/csu_ysy/dataset/train2014',
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default="/home/csu_ysy/dataset/wikiart/",
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='model/vgg_normalised.pth')
parser.add_argument('--sample_path', type=str, default='samples+safin(new)', help='Derectory to save the intermediate samples')

# training options
parser.add_argument('--resume', default='iter149999_glow.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--save_dir', default='./experiments+safin(s18c3_w/oGAN)',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-6)
parser.add_argument('--start_iter', type=int, default=0)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--style_weight', type=float, default=18)
parser.add_argument('--sup_weight', type=float, default=0.1)
parser.add_argument('--content_weight', type=float, default=3)
parser.add_argument('--contrastive_weight_c', type=float, default=0.5)
parser.add_argument('--contrastive_weight_s', type=float, default=1)
parser.add_argument('--gan_weight', type=float, default=0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--n_flow', default=8, type=int, help='number of flows in each block')# 32
parser.add_argument('--n_block', default=2, type=int, help='number of blocks')# 4
parser.add_argument('--no_lu', action='store_true', help='use plain convolution instead of LU decomposed version')
parser.add_argument('--affine', default=False, type=bool, help='use affine coupling instead of additive')
parser.add_argument('--operator', type=str, default='adain',
                    help='style feature transfer operator')

args = parser.parse_args('')


device = torch.device("cuda")

save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
#writer = SummaryWriter(log_dir=str(log_dir))
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
args.resume = os.path.join(args.save_dir, args.resume)

vgg = net.vgg

valid = 1
fake = 0
D = net.MultiDiscriminator()
D.to(device)

vgg.load_state_dict(torch.load(args.vgg))
#vgg = nn.Sequential(*list(vgg.children())[:44])

network = net.Net(vgg,8, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)
if args.resume:
    if os.path.isfile(args.resume):
        print("--------loading checkpoint----------")
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_iter = checkpoint['iter']
        network.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("--------no checkpoint found---------")
network.train()
network.to(device)
network = nn.DataParallel(network, device_ids=[0])

content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=int(args.batch_size),
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=int(args.batch_size),
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam(network.module.parameters(), lr=args.lr)
optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))


for i in tqdm(range(args.start_iter, args.max_iter)):

    adjust_learning_rate(optimizer, iteration_count=i)
    adjust_learning_rate(optimizer_D, iteration_count=i)
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)

    ######################################################
   # content_images_ = content_images[1:]
   # content_images_ = torch.cat([content_images_, content_images[0:1]], 0)
   # content_images = torch.cat([content_images, content_images_], 0)
   # style_images = torch.cat([style_images, style_images], 0)
    ######################################################
   
    img, loss_c, loss_s,loss_sup= network(content_images,False, style_images)

    # train discriminator
    loss_gan_d = D.compute_loss(style_images, valid) + D.compute_loss(img.detach(), fake)
    optimizer_D.zero_grad()
    loss_gan_d.backward()
    optimizer_D.step()
    
    # train generator
    loss_sup= args.sup_weight*loss_sup
    #loss_supc=args.sup_weight*loss_supc
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    #loss_contrastive_c = args.contrastive_weight_c * loss_contrastive_c
    #loss_contrastive_s = args.contrastive_weight_s * loss_contrastive_s
    loss_gan_g = args.gan_weight * D.compute_loss(img, valid)
    
    loss = loss_c+ loss_s+ loss_gan_g+loss_sup

    
    loss.sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    

    ############################################################################
    output_dir = Path(args.sample_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    if   ((i + 1) % 500 == 0):
        output = torch.cat([style_images, content_images, img], 2)
        output_name = output_dir / 'output{:d}.jpg'.format(i + 1)
        save_image(output, str(output_name))
        print(loss.sum().cpu().detach().numpy(),"-lc:",loss_c.sum().cpu().detach().numpy(),"-ls:",loss_s.sum().cpu().detach().numpy(),
              "-lsup:",loss_sup.sum().cpu().detach().numpy()
              )
              
    ############################################################################

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter or i ==1:
       
        state_dict = network.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))

        state = {'iter': i, 'state_dict': state_dict, 'optimizer': optimizer.state_dict()}
        torch.save(state,'./experiments+safin(s18c3_w/oGAN)/iter{}_glow.pth'.format(i))
