# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import time
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from miscc.utils import build_super_images, build_super_images2
from models.losses import get_loss
from miscc.config import cfg
from datasets import TextDataset
from datasets import prepare_data
from model import CNN_ENCODER, RNN_ENCODER
from models.bigGAN_deep import Generator, Discriminator

if torch.cuda.is_available():
    device = torch.device("cuda")

torch.cuda.set_device(cfg.GPU_ID)

num_epochs = 500
branch_num = 3
batch_size = 6
lr = 0.0001
sample_dir = '../BigGan_sample'
d_steps_my = 1
lambda_gp = 10

if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

G_net = Generator().to(device)
D_net = Discriminator().to(device)


g_optimizers = optim.Adam(G_net.parameters(), lr=0.0001, betas=(0., 0.999))
d_optimizers = optim.Adam(D_net.parameters(), lr=0.0004, betas=(0., 0.999))

def reset_gard():
    g_optimizers.zero_grad()
    d_optimizers.zero_grad()

def denorm(x):
    out = (x + 1)/2
    return out.clamp(0,1)

def build_models(n_words):
    # ###################encoders######################################## #
    if cfg.TRAIN.NET_E == '':
        print('Error: no pretrained text-image encoders')
        return

    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
    state_dict = \
        torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
    image_encoder.load_state_dict(state_dict)
    for p in image_encoder.parameters():
        p.requires_grad = False
    print('Load image encoder from:', img_encoder_path)
    image_encoder.eval()

    text_encoder = \
        RNN_ENCODER(n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = \
        torch.load(cfg.TRAIN.NET_E,
                   map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    for p in text_encoder.parameters():
        p.requires_grad = False
    print('Load text encoder from:', cfg.TRAIN.NET_E)
    text_encoder.eval()

    # ########################################################### #
    if cfg.CUDA:
        text_encoder = text_encoder.cuda()
        image_encoder = image_encoder.cuda()

    return [text_encoder, image_encoder]

# Get data loader
imsize = cfg.TREE.BASE_SIZE * (2 ** (branch_num - 1))   # 64* (2 ** (3-1))
# image_transform = transforms.Compose([
#     transforms.Resize(int(imsize * 76 / 64)),
#     transforms.RandomCrop(imsize),
#     transforms.RandomHorizontalFlip()])

image_transform = transforms.Compose([#transforms.RandomResizedCrop(256,
                                #scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                            transforms.Resize(int(imsize * 76 / 64)),
                            transforms.RandomCrop(256),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Lambda(lambda img: (img * 2) - 1)])

dataset = TextDataset(cfg.DATA_DIR, 'train',
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)
assert dataset
dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

loss = get_loss('wgangp', device)
text_encoder, image_encoder = build_models(dataset.n_words)
text_encoder, image_encoder = text_encoder.to(device), image_encoder.to(device)

tb = SummaryWriter()

for epoch in range(num_epochs):
    start_t = time.time()
    d_steps = 0
    for data in dataloader:
        imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

        hidden = text_encoder.init_hidden(batch_size)
        words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
        # words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

        # D_net.zero_grad()
        with torch.no_grad():
            noise = torch.randn(batch_size, 128, dtype=torch.float32, device=device)
            fake_images = G_net(noise, sent_emb)

        #################### update D network ###############################
        d_logits_real = D_net(imgs, sent_emb)
        d_loss_real = - d_logits_real.mean()
        # d_loss_real.backward(retain_graph=True)

        d_logits_fake = D_net(fake_images, sent_emb)
        d_loss_fake = d_logits_fake.mean()
        # d_loss_fake.backward(retain_graph=True)

        d_loss = d_loss_real + d_loss_fake
        reset_gard()
        d_loss.backward()
        d_optimizers.step()

        ###############################################
        # d_loss_gp, retain_graph = loss.loss_d_additional(D_net, imgs, fake_images, sent_emb)
        # d_loss_gp.backward(retain_graph=False)
        # d_loss = d_loss_real.item() + d_loss_fake.item() + d_loss_gp.item()
        # d_loss += d_loss_gp

        # reset_gard()
        # d_loss.backward(retain_graph=False)
        # d_optimizers.step()

        ############################# update G network ##############################
        G_net.zero_grad()
        noise = torch.randn(batch_size, 128, dtype=torch.float32, device=device)
        fake_image = G_net(noise, sent_emb)

        g_logits_fake = D_net(fake_image, sent_emb)
        g_loss = - g_logits_fake.mean()

        reset_gard()
        g_loss.backward()
        g_optimizers.step()


        tb.add_scalar('g_loss', g_loss, epoch)
        tb.add_scalar('d_loss', d_loss, epoch)

    for name_g, param_g in G_net.named_parameters():
        tb.add_histogram(name_g, param_g, epoch)
        tb.add_histogram(f'{name_g}.grad', param_g.grad, epoch)
    for name_d, param_d in D_net.named_parameters():
        tb.add_histogram(name_d, param_d, epoch)
        tb.add_histogram(f'{name_d}', param_d.grad, epoch)

    end_t = time.time()
    print('epoch:{}, G_loss:{}, D_loss:{}, time:{}'.format(epoch, g_loss.item(), d_loss, end_t-start_t))

    if (epoch + 1) == 1:  # 只是保存一张
        images = imgs.data
        save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))

    fake_images = fake_images.data
    save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch + 1)))

tb.close()
torch.save(G_net.state_dict(), 'G_net.ckpt')
torch.save(D_net.state_dict(), 'D_net.ckpt')
