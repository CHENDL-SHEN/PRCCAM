import cv2
import random
import numpy as np

from PIL import Image

def get_strided_size(orig_size, stride):
    return ((orig_size[0]-1)//stride+1, (orig_size[1]-1)//stride+1)  # 正卷积尺寸计算公式

def get_strided_up_size(orig_size, stride):
    strided_size = get_strided_size(orig_size, stride)
    return strided_size[0]*stride, strided_size[1]*stride

def imshow(image, delay=0, mode='RGB', title='show'):
    if mode == 'RGB':
        demo_image = image[..., ::-1]
    else:
        demo_image = image

    cv2.imshow(title, demo_image)
    if delay >= 0:
        cv2.waitKey(delay)

def transpose(image):
    return image.transpose((1, 2, 0))

def denormalize(image, mean=None, std=None, dtype=np.uint8, tp=True):
    if tp:
        image = transpose(image)
        
    if mean is not None:
        image = (image * std) + mean
    
    if dtype == np.uint8:
        image *= 255.
        return image.astype(np.uint8)
    else:
        return image

def colormap(cam, shape=None, mode=cv2.COLORMAP_JET):
    if shape is not None:
        h, w, c = shape
        cam = cv2.resize(cam, (w, h))
    cam = cv2.applyColorMap(cam, mode)
    return cam

def decode_from_colormap(data, colors):
    ignore = (data == 255).astype(np.int32)

    mask = 1 - ignore
    data *= mask

    h, w = data.shape
    image = colors[data.reshape((h * w))].reshape((h, w, 3))

    ignore = np.concatenate([ignore[..., np.newaxis], ignore[..., np.newaxis], ignore[..., np.newaxis]], axis=-1)
    image[ignore.astype(np.bool)] = 255
    return image

def normalize(cam, epsilon=1e-5):
    cam = np.maximum(cam, 0)
    max_value = np.max(cam, axis=(0, 1), keepdims=True)
    return np.maximum(cam - epsilon, 0) / (max_value + epsilon)


def crf_inference(img, probs, t=10, scale_factor=1, labels=6):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=1 / scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=67 / scale_factor, srgb=3, rgbim=np.copy(img), compat=4)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))

def crf_with_alpha_un(ori_image, cams, keys):
    # h, w, c -> c, h, w
    # cams = cams.transpose((2, 0, 1))

    # bg_score = np.power(1 - np.max(cams, axis=0, keepdims=True), alpha)
    # bgcam_score = np.concatenate((bg_score, cams), axis=0)
    bgcam_score = cams
    cams_with_crf = crf_inference(ori_image, bgcam_score, labels=bgcam_score.shape[0])
    # return cams_with_crf.transpose((1, 2, 0))
    n_crf_al = dict()
    for i, key in enumerate(keys):
        n_crf_al[key] = cams_with_crf[i]
    return cams_with_crf

def crf_with_alpha(ori_image, cams):
    # h, w, c -> c, h, w
    # cams = cams.transpose((2, 0, 1))

    # bg_score = np.power(1 - np.max(cams, axis=0, keepdims=True), alpha)
    # bgcam_score = np.concatenate((bg_score, cams), axis=0)
    bgcam_score = cams['hr_cam']
    cams_with_crf = crf_inference(ori_image, bgcam_score, labels=bgcam_score.shape[0])
    # return cams_with_crf.transpose((1, 2, 0))
    keys = cams['keys']
    n_crf_al = dict()
    for i, key in enumerate(keys):
        n_crf_al[key] = cams_with_crf[i]
    return cams_with_crf


def crf_inference_label(img, labels, t=10, n_labels=21, gt_prob=0.7):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels
    h, w = img.shape[:2]

    d = dcrf.DenseCRF2D(w, h, n_labels) # 使用densecrf类
    # d.setUnaryEnergy(-np.log(labels).reshape((n_labels, -1)).astype(np.float32))
    # unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)
    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=True)  # 如果“True”，则将标签值“0”视为“可能是任何东西”，即具有此值的项将得到一致的一元概率，不将其当作标签。如果“False”，不要特别对待值“0”，而是像对待任何其他类一样对待它。
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=1, compat=3)
    d.addPairwiseBilateral(sxy=67, srgb=3, rgbim=np.ascontiguousarray(np.copy(img)), compat=4)

    q = d.inference(t)

    return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)
