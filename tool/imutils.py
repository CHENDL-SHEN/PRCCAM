
import PIL.Image
import random
import numpy as np

class RandomResizeLong():

    def __init__(self, min_long, max_long):
        self.min_long = min_long
        self.max_long = max_long

    def __call__(self, img):

        target_long = random.randint(self.min_long, self.max_long)
        w, h = img.size

        if w < h:
            target_shape = (int(round(w * target_long / h)), target_long)
        else:
            target_shape = (target_long, int(round(h * target_long / w)))

        img = img.resize(target_shape, resample=PIL.Image.CUBIC)

        return img


class RandomCrop():

    def __init__(self, cropsize):
        self.cropsize = cropsize

    def __call__(self, imgarr):

        h, w, c = imgarr.shape

        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)

        w_space = w - self.cropsize
        h_space = h - self.cropsize

        if w_space > 0:
            cont_left = 0
            img_left = random.randrange(w_space+1)
        else:
            cont_left = random.randrange(-w_space+1)
            img_left = 0

        if h_space > 0:
            cont_top = 0
            img_top = random.randrange(h_space+1)
        else:
            cont_top = random.randrange(-h_space+1)
            img_top = 0

        container = np.zeros((self.cropsize, self.cropsize, imgarr.shape[-1]), np.float32)
        container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
            imgarr[img_top:img_top+ch, img_left:img_left+cw]

        return container

def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw

def crop_with_box(img, box):
    if len(img.shape) == 3:
        img_cont = np.zeros((max(box[1]-box[0], box[4]-box[5]), max(box[3]-box[2], box[7]-box[6]), img.shape[-1]), dtype=img.dtype)
    else:
        img_cont = np.zeros((max(box[1] - box[0], box[4] - box[5]), max(box[3] - box[2], box[7] - box[6])), dtype=img.dtype)
    img_cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
    return img_cont


def random_crop(images, cropsize, fills):
    if isinstance(images[0], PIL.Image.Image):
        imgsize = images[0].size[::-1]
    else:
        imgsize = images[0].shape[:2]
    box = get_random_crop_box(imgsize, cropsize)

    new_images = []
    for img, f in zip(images, fills):

        if isinstance(img, PIL.Image.Image):
            img = img.crop((box[6], box[4], box[7], box[5]))
            cont = PIL.Image.new(img.mode, (cropsize, cropsize))
            cont.paste(img, (box[2], box[0]))
            new_images.append(cont)

        else:
            if len(img.shape) == 3:
                cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*f
            else:
                cont = np.ones((cropsize, cropsize), img.dtype)*f
            cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
            new_images.append(cont)

    return new_images


class AvgPool2d():

    def __init__(self, ksize):
        self.ksize = ksize

    def __call__(self, img):
        import skimage.measure

        return skimage.measure.block_reduce(img, (self.ksize, self.ksize, 1), np.mean)


class RandomHorizontalFlip():
    def __init__(self):
        return

    def __call__(self, img):
        if bool(random.getrandbits(1)):
            img = np.fliplr(img).copy()
        return img


class CenterCrop():

    def __init__(self, cropsize, default_value=0):
        self.cropsize = cropsize
        self.default_value = default_value

    def __call__(self, npimg):

        h, w = npimg.shape[:2]

        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)

        sh = h - self.cropsize
        sw = w - self.cropsize

        if sw > 0:
            cont_left = 0
            img_left = int(round(sw / 2))
        else:
            cont_left = int(round(-sw / 2))
            img_left = 0

        if sh > 0:
            cont_top = 0
            img_top = int(round(sh / 2))
        else:
            cont_top = int(round(-sh / 2))
            img_top = 0

        if len(npimg.shape) == 2:
            container = np.ones((self.cropsize, self.cropsize), npimg.dtype)*self.default_value
        else:
            container = np.ones((self.cropsize, self.cropsize, npimg.shape[2]), npimg.dtype)*self.default_value

        container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
            npimg[img_top:img_top+ch, img_left:img_left+cw]

        return container


def HWC_to_CHW(img):

    # transpose()函数的作用就是调换数组的行列值的索引值，类似于求矩阵的转置；第二个参数如(2, 0, 1)就是改变索引值的地方
    # 正常的数组索引值应该是（0，1，2）对应着（x，y，z），而transpose()函数改变了img的索引值，如下变成了(2, 0, 1)这就是把z的位置放在第一个，所以索引值就变成了（z，x，y）
    # 这样造成的效果是，比如访问img这个多维数组中的某个元素
    return np.transpose(img, (2, 0, 1))


class RescaleNearest():
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, npimg):
        import cv2
        return cv2.resize(npimg, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)




def crf_inference(img, probs, t=10, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    # 传原始三通道图像img进来就是想得到它的形状
    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    # 将softmax类概率转换为一元势函数（这应该是条件随机场CRF里的概念）
    # 即我们之前先对图片使用训练好的网络预测得到最终经过softmax函数得到的分类结果，这里需要将这个结果转成一元势
    unary = unary_from_softmax(probs)

    # np.ascontiguousarray将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。
    unary = np.ascontiguousarray(unary)

    # 构建好一元势后需要调用：d.setUnaryEnergy()将该一元势添加到CRF中
    d.setUnaryEnergy(unary)

    # 这里是创建图像的二元势，用来描述一对像素对之间的关系
    # 二维情况下，增加最常见的二元势有两种实用方法:二元势即用于描述像素点和像素点之间的关系，鼓励相似像素分配相同的标签，而相差较大的像素分配不同的标签。
    # 这个相似的定义与颜色值srgb和实际相对距离sxy相关，所以CRF能够使图片尽量在边界处分割。
    # d.addPairwiseGaussian这个函数创建的是颜色无关特征，这里只有位置特征(只有参数实际相对距离sxy)，并添加到CRF中
    # d.addPairwiseBilateral这个函数根据原始图像img创建颜色相关和位置相关特征并添加到CRF中，特征为(x,y,r,g,b)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)

    # d.inference（）将一元势和二元势结合起来就能够比较全面地去考量像素之间的关系，并得出优化后的结果，用t次迭代进行推理最简单的方法是:
    Q = d.inference(t)

    # Q是一个包装好的特征矩阵。本项目的特征包装器实现缓冲接口，可以简单地转换为numpy数组np.array(Q)
    # MAP预测是：map = np.argmax(Q, axis=0).reshape((640,480))
    # 这里有点疑惑，不应该是np.argmax(Q, axis=0).reshape((640,480))，之后将Q变成一个numpy的多维数组，怎么这里是变成了np.array(Q).reshape
    return np.array(Q).reshape((n_labels, h, w))