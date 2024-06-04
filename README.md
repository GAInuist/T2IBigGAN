In recent years, Generative Adversarial Networks (GANs) have become the mainstream Text-to-Image(T2I) framework. However, a standard normal distribution noise of inputs cannot provide enough information to ynthesize an image that approaches the ground-truth image distribution. Moreover, the multi-stage generation strategy makes it complex to apply T2I. Therefore, this paper proposes a novel feature-grounded single-stage T2I model, which takes the ”real” distribution learned from training images as one input and introduces a worst-caseoptimized similarity measure into the loss function to enhance the generation capacity. Experimental results on two benchmark datasets demonstrate that the proposed model has competitive performance on FID and IS compared to some classical and state-of-the-art works and improves the similarity between the generated image, the text, and the ground truth.


文件code中保存了代码

文件DAMSMenconders中保存各个数据集对应的预训练模型，比如bird/image_encoder200.pth 或者 bird/text_encoder200.pth

Models文件里放的是BigGAN的模型代码，根据需要选择biggan模型或者biggandeep模型进行训练

Data文件中保存的是训练的数据集，bird或者ms_coco数据集
