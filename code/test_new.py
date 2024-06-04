
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

    # load image encoder
    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
    state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
    image_encoder.load_state_dict(state_dict)
    print('Load image encoder from:', img_encoder_path)
    image_encoder = image_encoder.cuda()
    image_encoder.eval()

    nz = cfg.GAN.Z_DIM
    noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
    noise = noise.cuda()
    # noise = torch.randn(batch_size, 128, dtype=torch.float32, device=device)

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
    # save_real_dir = '%s/%s' % (s_tmp, real_dir)
    # mkdir_p(save_real_dir)

    cnt = 0
    R_count = 0
    R = np.zeros(30000)
    cont = True
    for ii in range(11):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
        if (cont == False):
            break
        for step, data in enumerate(dataloader, 0):
            cnt += batch_size
            if (cont == False):
                break
            if step % 100 == 0:
                print('cnt: ', cnt)
            # if step > 50:
            #     break

            imgs, image, captions, cap_lens, class_ids, keys = prepare_data(data)

            hidden = text_encoder.init_hidden(batch_size)
            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            mask = (captions == 0)
            num_words = words_embs.size(2)
            if mask.size(1) > num_words:
                mask = mask[:, :num_words]

            #######################################################
            # (2) Generate fake images
            ######################################################
            noise.data.normal_(0, 1)
            # fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask, cap_lens)
            fake_imgs = netG(noise, sent_emb, words_embs)
            for j in range(batch_size):
                s_tmp = '%s/single/%s' % (save_dir, keys[j])
                folder = s_tmp[:s_tmp.rfind('/')]
                if not os.path.isdir(folder):
                    # print('Make a new folder: ', folder)
                    mkdir_p(folder)
                k = -1
                # for k in range(len(fake_imgs)):
                im = fake_imgs[j].data.cpu().numpy()
                # [-1, 1] --> [0, 255]
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)
                fullpath = '%s_s%d_%d.png' % (s_tmp, k, ii)
                im.save(fullpath)

            _, cnn_code = image_encoder(fake_imgs)

            for i in range(batch_size):
                mis_captions, mis_captions_len = dataset.get_mis_caption(class_ids[i])
                hidden = text_encoder.init_hidden(99)
                _, sent_emb_t = text_encoder(mis_captions, mis_captions_len, hidden)
                rnn_code = torch.cat((sent_emb[i, :].unsqueeze(0), sent_emb_t), 0)
                ### cnn_code = 1 * nef
                ### rnn_code = 100 * nef
                scores = torch.mm(cnn_code[i].unsqueeze(0), rnn_code.transpose(0, 1))  # 1* 100
                cnn_code_norm = torch.norm(cnn_code[i].unsqueeze(0), 2, dim=1, keepdim=True)
                rnn_code_norm = torch.norm(rnn_code, 2, dim=1, keepdim=True)
                norm = torch.mm(cnn_code_norm, rnn_code_norm.transpose(0, 1))
                scores0 = scores / norm.clamp(min=1e-8)
                if torch.argmax(scores0) == 0:
                    R[R_count] = 1
                R_count += 1

            if R_count >= 30000:
                sum = np.zeros(10)
                np.random.shuffle(R)
                for i in range(10):
                    sum[i] = np.average(R[i * 3000:(i + 1) * 3000 - 1])
                R_mean = np.average(sum)
                R_std = np.std(sum)
                print("R mean:{:.4f} std:{:.4f}".format(R_mean, R_std))
                cont = False

