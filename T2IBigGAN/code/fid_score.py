import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d


from inception import InceptionV3
import os
os.environ['CUDA_VISIBLE_DEVICE'] = '0'
torch.cuda.set_device(0)

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=16,
                    help='Batch size to use')
parser.add_argument('--num-workers', type=int, default=0,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--device', type=str, default='cuda',
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('--path', type=str, default=None, nargs=2,
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))


def get_activations(pred, dims=2048):

    pred_arr = np.empty((20, dims))   # 全0数组

    start_idx = 0
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
    # if pred.size(2) != 1 or pred.size(3) != 1:
    #     pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
    #     print('1111111111111111111111111111111111')
    # print(pred)
    pred = pred.squeeze(3).squeeze(2).cpu().numpy()
    # pred = pred.squeeze(3).squeeze(2)


    pred_arr[start_idx:start_idx + pred.shape[0]] = pred

    # start_idx = start_idx + pred.shape[0]

    return pred_arr

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    # covmean = tf.linalg.sqrtm(sigma1.dot(sigma2))


    # if not np.isfinite(covmean).all():
    #     msg = ('fid calculation produces singular product; '
    #            'adding %s to diagonal of cov estimates') % eps
    #     print(msg)
    #     print('+++++++++++++++++++++++++++++++++++++++++')
    #     offset = np.eye(sigma1.shape[0]) * eps
    #     covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):

            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
        # print(covmean)

    tr_covmean = np.trace(covmean)


    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def calculate_activation_statistics(pred, dims=2048):

    act = get_activations(pred, dims)
    mu = np.mean(act, axis=0)

    sigma = np.cov(act, rowvar=False)

    return mu, sigma

def compute_statistics_of_path(pred, dims):

    m, s = calculate_activation_statistics(pred, dims)

    return m, s

def calculate_fid_given_paths(pred_real, pred_fake, dims):
    """Calculates the FID of two paths"""

    m1, s1 = compute_statistics_of_path(pred_real,
                                        dims)
    m2, s2 = compute_statistics_of_path(pred_fake,
                                        dims)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value

def img_score(pred_real, pred_fake):
    args = parser.parse_args()

    fid_value = calculate_fid_given_paths(pred_real, pred_fake,
                                          args.dims
                                          )

    return fid_value

