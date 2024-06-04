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
import torch.nn.functional as F

from miscc.utils import build_super_images, build_super_images2
from models.losses import get_loss
from miscc.config import cfg
from datasets import TextDataset
from datasets import prepare_data
from model import CNN_ENCODER, RNN_ENCODER
from models.bigGAN_deep import Generator, Discriminator

# from pytorch_fid.inception import InceptionV3
from fid_score import img_score
import clip


if torch.cuda.is_available():
    device = torch.device("cuda")

torch.cuda.set_device(cfg.GPU_ID)

num_epochs = 701
branch_num = 3
batch_size = 16
lr = 0.0001
sample_dir = '../BigGan_sample'
d_steps_my = 1
step_k = 1
thta = 0.01

if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

G_net = Generator().to(device)
D_net = Discriminator().to(device)

# block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
# Incep_model = InceptionV3([block_idx]).to(device)

model, preprocess = clip.load("ViT-B/32", device=device)

g_optimizers = optim.Adam(G_net.parameters(), lr=0.0001, betas=(0., 0.999))
d_optimizers = optim.Adam(D_net.parameters(), lr=0.0004, betas=(0., 0.999))

def reset_grad():
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
image_transform = transforms.Compose([
                            transforms.Resize(int(imsize * 76 / 64)),
                            transforms.RandomCrop(imsize),
                            transforms.RandomHorizontalFlip()])

image_transforms = transforms.Compose([#transforms.RandomResizedCrop(256,
                                #scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                            transforms.Resize(int(imsize * 76 / 64)),
                            transforms.RandomCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Lambda(lambda img: (img * 2) - 1)])

transforms_fake = transforms.ToPILImage()

dataset = TextDataset(cfg.DATA_DIR, 'train',
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)
assert dataset
dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

loss = get_loss('hingegan', device)
text_encoder, image_encoder = build_models(dataset.n_words)
text_encoder, image_encoder = text_encoder.to(device), image_encoder.to(device)

tb = SummaryWriter()

for epoch in range(num_epochs):
    start_t = time.time()
    d_steps = 0
    for data in dataloader:
        # imgs, captions, class_ids, keys = prepare_data(data)
        imgs, image, captions, cap_lens, class_ids, keys = prepare_data(data)
        # print('iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii')
        # print(imgs[0].shape)
        # text = clip.tokenize(captions).to(device)
        sent_embedding, words_embeddings = model.encode_text(captions)
        sent_embedding, words_embeddings = sent_embedding.detach(), words_embeddings.detach()

        image = image.cuda()
        # sent_feature = F.normalize(sent_emb)
        imgs_feature = model.encode_image(image)
        # print('1111111111111111111111111111111111111111111')
        # print(sent_emb)
        # print(sent_emb.shape)
        # print(sent_feature.shape)
        # print(imgs_feature.shape)
        # imgs_features = F.normalize(imgs_feature)
        # hidden = text_encoder.init_hidden(batch_size)
        # words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
        # words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

        # D_net.zero_grad()
        with torch.no_grad():
            noise = torch.randn(batch_size, 128, dtype=torch.float32, device=device)
            # fake_images, mu, logvar = G_net(noise, sent_emb, words_embs)
            fake_images = G_net(noise, sent_embedding)
        #################### update D network ##############################
        # print(fake_images.shape)
        # d_logits_real, _ = D_net(imgs[0], sent_emb)
        d_logits_real = D_net(imgs[0], sent_embedding)
        d_loss_real = torch.nn.ReLU()(1.0 - d_logits_real).mean()
        # d_loss_real, retain_graph = loss.loss_d_real(d_logits_real, None)
        # d_loss_real.backward()

        # d_logits_fake, _ = D_net(fake_images, sent_emb)
        d_logits_fake = D_net(fake_images, sent_embedding)

        d_loss_fake = torch.nn.ReLU()(1.0 + d_logits_fake).mean()
        # d_loss_fake, retain_graph = loss.loss_d_fake(d_logits_fake, None)
        # d_loss_fake.backward()

        # pred_real = Incep_model(imgs)[0]
        # pred_fake = Incep_model(fake_images)[0]
        # print(d_loss_real)
        # print(d_loss_fake)
        d_loss = d_loss_real + d_loss_fake
        # print('----------------------------------------------')
        # print(d_loss_fake)
        # print(d_loss_real)
        # print(d_loss)

        reset_grad()
        d_loss.backward()
        d_optimizers.step()

        ######################### updata G network #######################
        # for d in D_net.parameters():
        #     d.requires_grad = False
        # for g in G_net.parameters():
        #     g.requires_grad = True
        d_steps += 1
        if d_steps < step_k:
            update_g = False
        else:
            update_g = True
            d_steps = 0

        if update_g:
            # G_net.zero_grad()
            # with torch.no_grad():
            noise = torch.randn(batch_size, 128, dtype=torch.float32, device=device)
            # fake_images, mu, logvar = G_net(noise, sent_emb, words_embs)
            fake_images = G_net(noise, sent_embedding)

            # g_logits_fake, _ = D_net(fake_images, sent_emb)
            g_logits_fake = D_net(fake_images, sent_embedding)
            g_loss_fake = - g_logits_fake.mean()
            # g_loss, retain_graph = loss.loss_g(g_logits_fake, None)

            fake_list = []
            for b_i in range(fake_images.shape[0]):
                fake_im = transforms_fake(fake_images[b_i].cpu())
                fake_im = image_transforms(fake_im)
                fake_list.append(fake_im)
            fakeimages = torch.stack(fake_list)
            # print('0-0-0-0-0-0-0-0-0-0--0-0-0-0-0-0-0--0-0-')
            # print(fakeimages.shape)
            fakeimages = fakeimages.cuda()
            fakeimages_features = model.encode_image(fakeimages)
            # print(fakeimages_features.shape)
            # fakeimages_features = F.normalize(fakeimages_features)
            # cos_fakeI_I = imgs_features.mm(fakeimages_features.t())
            # cos_fakeI_T = sent_feature.mm(fakeimages_features.t())
            # cos_I_T = sent_feature.mm(imgs_features.t())
            cos_fakeI_I = F.cosine_similarity(imgs_feature, fakeimages_features)
            cos_fakeI_T = F.cosine_similarity(sent_embedding, fakeimages_features)
            cos_I_T = F.cosine_similarity(sent_embedding, imgs_feature)
            # print('cos_fakeI_I:{}, cos_fakeI_T:{}, cos_I_T:{}'.format(cos_fakeI_I,cos_fakeI_T,cos_I_T))
            cos = torch.stack([cos_I_T, cos_fakeI_T, cos_fakeI_I], dim=0)
            # print('--------------------------------------------')
            # print(cos)
            max_cos, index = torch.max(cos, dim=0)
            # print(max_cos)
            # print(max_cos.values)
            loss_cos = - max_cos.mean()
            # print(loss_cos)
            g_loss = g_loss_fake + loss_cos

            reset_grad()
            g_loss.backward()
            g_optimizers.step()

            tb.add_scalar('g_loss', g_loss, epoch)
            tb.add_scalar('d_loss', d_loss, epoch)

    for name, weight in G_net.named_parameters():
        tb.add_histogram(name, weight, epoch)
        tb.add_histogram(f'{name}.grad', weight.grad, epoch)
    for name, weight in D_net.named_parameters():
        tb.add_histogram(name, weight, epoch)
        tb.add_histogram(f'{name}.grad', weight.grad, epoch)

    end_t = time.time()
    print('epoch:{}, G_loss:{}, D_loss:{}, time:{}'.format(epoch, g_loss.item(), d_loss.item(), end_t-start_t))

    if (epoch + 1) == 1:  # 只是保存一张
        images = imgs[0].data
        save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))

    save_image(denorm(fake_images.data), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch + 1)))

    if epoch % 100 == 0:
        torch.save(G_net.state_dict(), 'G_net_{}.ckpt'.format(epoch))
        torch.save(D_net.state_dict(), 'D_net_{}.ckpt'.format(epoch))

tb.close()