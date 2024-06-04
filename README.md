# Feature-Grounded Single-Stage Text-to-Image Generation

In recent years, Generative Adversarial Networks (GANs) have become the mainstream Text-to-Image(T2I) framework. However, a standard normal distribution noise of inputs cannot provide enough information to ynthesize an image that approaches the ground-truth image distribution. Moreover, the multi-stage generation strategy makes it complex to apply T2I. Therefore, this paper proposes a novel feature-grounded single-stage T2I model, which takes the ”real” distribution learned from training images as one input and introduces a worst-caseoptimized similarity measure into the loss function to enhance the generation capacity. Experimental results on two benchmark datasets demonstrate that the proposed model has competitive performance on FID and IS compared to some classical and state-of-the-art works and improves the similarity between the generated image, the text, and the ground truth.

Over Framework
![4372ab51550a53609ce7c8219e32cde](https://github.com/GAInuist/T2IBigGAN/assets/157414652/663c4ca4-d95f-4c78-aeb0-ec401430ccc1)

## 注：
文件code中保存了代码

文件DAMSMenconders中保存各个数据集对应的预训练模型，比如bird/image_encoder200.pth 或者 bird/text_encoder200.pth

Models文件里放的是BigGAN的模型代码，根据需要选择biggan模型或者biggandeep模型进行训练

Data文件中保存的是训练的数据集，bird或者ms_coco数据集

## 模型权重文件下载点：
model.ckpt: https://drive.google.com/file/d/1u9TVR4wTgXgOnWLTG3aFvyByfXFv2Se4/view?usp=drive_link

model.ckpt.meta: https://drive.google.com/file/d/1TaP7ven0C54SeRqR4f6Xn0pqd1F3hHy1/view?usp=drive_link

image_encoder200.pth： https://drive.google.com/file/d/1Ii4RQbrDvjZQAZaQLJPQE5fDbIxgUcfu/view?usp=drive_link

text_encoder200.pth: https://drive.google.com/file/d/1CSMJDDlIcMjWkrZFYpmzc6SdHQowcqZ3/view?usp=drive_link

## 存放位置

model.ckpt和model.ckpt.meta存放于T2IBigGAN\evalution\inception_finetuned_models文件夹中

image_encoder200.pth和text_encoder200.pth存放于T2IBigGAN\DAMSMencoders\bird文件夹中



