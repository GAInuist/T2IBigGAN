
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
import numpy as np
from miscc.utils import mkdir_p, weights_init

from miscc.utils import build_super_images, build_super_images2
from models.losses import get_loss
from miscc.config import cfg
from datasets import TextDataset
from datasets import prepare_data
from model import CNN_ENCODER, RNN_ENCODER
from models.bigGAN_deep import Generator, Discriminator

if torch.cuda.is_available():
    device = torch.device('cuda')

def denorm(x):
    out = (x + 1)/2
    return out.clamp(0,1)

split_dir = 'test'
branch_num = 3
batch_size = 10

imsize = cfg.TREE.BASE_SIZE * (2 ** (branch_num - 1))   # 64* (2 ** (3-1))
image_transform = transforms.Compose([
                            transforms.Resize(int(imsize * 76 / 64)),
                            transforms.RandomCrop(imsize),
                            transforms.RandomHorizontalFlip()])

dataset = TextDataset(cfg.DATA_DIR, 'test',
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)
assert dataset
dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=False, num_workers=int(cfg.WORKERS))


if cfg.TRAIN.NET_G == '':
    print('Error: the path for morels is not found!')
else:
    if split_dir == 'test':
        split_dir = 'valid'
    real_dir = 'real_images'
    # Build and load the generator
    netG = Generator()
    # netG.apply(weights_init)
    netG.cuda()
    netG.eval()
    #
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = \
        torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    print('Load text encoder from:', cfg.TRAIN.NET_E)
    text_encoder = text_encoder.cuda()
    text_encoder.eval()

    # nz = cfg.GAN.Z_DIM
    # noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
    # noise = noise.cuda()
    noise = torch.randn(batch_size, 128, dtype=torch.float32, device=device)

    model_dir = cfg.TRAIN.NET_G
    state_dict = \
        torch.load(model_dir, map_location=lambda storage, loc: storage)
    # state_dict = torch.load(cfg.TRAIN.NET_G)
    netG.load_state_dict(state_dict)
    print('Load G from: ', model_dir)

    # the path to save generated images
    s_tmp = model_dir[:model_dir.rfind('.ckpt')]
    save_dir = '%s/%s' % (s_tmp, split_dir)
    mkdir_p(save_dir)
    save_real_dir = '%s/%s' % (s_tmp, real_dir)
    mkdir_p(save_real_dir)

    cnt = 0
    count = 0

    for _ in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
        for step, data in enumerate(dataloader, 0):
            cnt += batch_size
            if step % 100 == 0:
                print('step: ', step)
            # if step > 50:
            #     break

            imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

            hidden = text_encoder.init_hidden(batch_size)
            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            # mask = (captions == 0)
            # num_words = words_embs.size(2)
            # if mask.size(1) > num_words:
            #     mask = mask[:, :num_words]

            #######################################################
            # (2) Generate fake images
            ######################################################
            noise.data.normal_(0, 1)
            fake_imgs= netG(noise, sent_emb)
            # fake_imgs = netG(noise, sent_emb, words_embs)
            # save_image(denorm(fake_imgs.data), os.path.join(save_dir, 'fake_images-{}.png'.format(step)))
            # save_image(denorm(imgs[2].data), os.path.join(save_dir, 'real_images-{}.png'.format(step)))

            for j in range(batch_size):
                save_name = '%s/%d.png' % (save_dir, count + j)
                im = fake_imgs[j].data.cpu().numpy()
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)
                im.save(save_name)

                save_rel_name = '%s/%d.png' % (save_real_dir, count + j)
                # real_im = imgs[2][j].data.cpu().numpy()
                # # real_im = denorm(real_im)
                # real_im = (real_im + 1.0) * 127.5
                # real_im = real_im.astype(np.uint8)
                # real_im = np.transpose(real_im, (1, 2, 0))
                # real_im = Image.fromarray(real_im)
                # real_im.save(save_rel_name)

                real_im = imgs[2][j].data
                save_image(denorm(real_im), save_rel_name)

            count += batch_size