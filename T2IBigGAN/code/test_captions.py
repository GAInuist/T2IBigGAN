# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import time
from torchvision.utils import save_image, make_grid
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

branch_num = 3

if torch.cuda.is_available():
    device = torch.device('cuda')

def denorm(x):
    out = (x + 1)/2
    return out.clamp(0,1)

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


def gen_example(data_dic, n_words):
    if cfg.TRAIN.NET_G == '':
        print('Error: the path for morels is not found!')
    else:
        # Build and load the generator
        text_encoder = \
            RNN_ENCODER(n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = \
            torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder = text_encoder.cuda()
        text_encoder.eval()

        # the path to save generated images
        netG = Generator()
        s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.ckpt')]
        model_dir = cfg.TRAIN.NET_G
        state_dict = \
            torch.load(model_dir, map_location=lambda storage, loc: storage)
        netG.load_state_dict(state_dict)
        print('Load G from: ', model_dir)
        netG.cuda()
        netG.eval()
        for key in data_dic:
            save_dir = '%s/%s' % (s_tmp, key)
            mkdir_p(save_dir)
            captions, cap_lens = data_dic[key]
            # print(captions)
            batch_size = captions.shape[0]
            nz = cfg.GAN.Z_DIM
            captions = Variable(torch.from_numpy(captions), volatile=True)
            cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

            captions = captions.cuda()
            cap_lens = cap_lens.cuda()

            for i in range(1):  # 16
                noise = torch.randn(batch_size, 128, dtype=torch.float32, device=device)
                #######################################################
                # (1) Extract text embeddings
                ######################################################
                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                # mask = (captions == 0)
                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                # fake_imgs = netG(noise, sent_emb)
                fake_imgs = netG(noise, sent_emb, words_embs)

                fake_imgs = make_grid(fake_imgs, 5, 0)
                save_image(denorm(fake_imgs.data), os.path.join(save_dir, 'fake_images-{}.png'.format(1)))



# def gen_example(wordtoix, algo):
# generate images from example sentences
from nltk.tokenize import RegexpTokenizer
#filepath = '%s/example_filenames.txt' % (cfg.DATA_DIR)
data_dic = {}
#with open(filepath, "r") as f:
#    filenames = f.read().decode('utf8').split('\n')
#    for name in filenames:
#        if len(name) == 0:
#            continue
# name = "self_test_captions"

name_dir = os.listdir(cfg.DATA_DIR + '/test_captions/')
for name in name_dir:
    filepath = '%s/test_captions/%s' % (cfg.DATA_DIR, name)
    with open(filepath, "r") as f:
        print('Load from:', name)
        sentences = f.read().encode('utf8').decode('utf8').split('\n')
        # a list of indices for a sentence
        captions = []
        cap_lens = []
        for sent in sentences:
            # print(sent)
            if len(sent) == 0:
                continue
            sent = sent.replace("\ufffd\ufffd", " ")
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(sent.lower())
            if len(tokens) == 0:
                print('sent', sent)
                continue
            rev = []
            for t in tokens:
                t = t.encode('ascii', 'ignore').decode('ascii')
                if len(t) > 0 and t in dataset.wordtoix:
                    rev.append(dataset.wordtoix[t])
            captions.append(rev)

            # print(captions)
            cap_lens.append(len(rev))

    max_len = np.max(cap_lens)
    sorted_indices = np.argsort(cap_lens)[::-1]
    cap_lens = np.asarray(cap_lens)
    cap_lens = cap_lens[sorted_indices]
    cap_array = np.zeros((len(captions), max_len), dtype='int64')
    for i in range(len(captions)):
        # idx = sorted_indices[i]
        cap = captions[i]
        c_len = len(cap)
        cap_array[i, :c_len] = cap
    key = name[(name.rfind('/') + 1):]
    data_dic[key] = [cap_array, cap_lens]
    gen_example(data_dic, dataset.n_words)


