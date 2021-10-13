import math
import random

import numpy
import onnx
import torch
import torch.backends.cudnn
from PIL import Image, ImageOps, ImageEnhance
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace, model_helper
from caffe2.python.onnx.backend import Caffe2Backend
from torchvision.transforms import functional


def set_seeds(seed=0):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def torch2onnx(model, shape):
    inputs = torch.randn(shape, requires_grad=True)
    model(inputs)

    dynamic_axes = {'input0': {0: 'batch'}, 'output0': {0: 'batch'}}

    _ = torch.onnx.export(model, inputs, 'weights/model.onnx',
                          export_params=True,
                          verbose=False,
                          input_names=["input0"],
                          output_names=["output0"],
                          keep_initializers_as_inputs=True,
                          dynamic_axes=dynamic_axes,
                          opset_version=10)

    onnx.checker.check_model(onnx.load('weights/model.onnx'))


def onnx2caffe():
    onnx_model = onnx.load('weights/model.onnx')
    caffe2_init, caffe2_predict = Caffe2Backend.onnx_graph_to_caffe2_net(onnx_model)
    caffe2_init_str = caffe2_init.SerializeToString()
    with open('weights/model.init.pb', "wb") as f:
        f.write(caffe2_init_str)
    caffe2_predict_str = caffe2_predict.SerializeToString()
    with open('weights/model.predict.pb', "wb") as f:
        f.write(caffe2_predict_str)


def print_benchmark(shape):
    model = model_helper.ModelHelper(name="model", init_params=False)

    init_net_proto = caffe2_pb2.NetDef()
    with open('weights/model.init.pb', "rb") as f:
        init_net_proto.ParseFromString(f.read())
    model.param_init_net = core.Net(init_net_proto)

    predict_net_proto = caffe2_pb2.NetDef()
    with open('weights/model.predict.pb', "rb") as f:
        predict_net_proto.ParseFromString(f.read())
    model.net = core.Net(predict_net_proto)

    model.param_init_net.GaussianFill([],
                                      model.net.external_inputs[0].GetUnscopedName(),
                                      shape=shape,
                                      mean=0.0,
                                      std=1.0)
    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)
    workspace.BenchmarkNet(model.net.Proto().name, 5, 100, True)


def add_weight_decay(model, weight_decay=1e-5):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]


def accuracy(output, target, top_k):
    with torch.no_grad():
        #max_k = max(top_k)
        max_k = 1
        batch_size = target.size(0)

        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in range(top_k):
            #correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            correct_k = correct[:(k+1)].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def copy_weights():
    with torch.no_grad():
        e_std = model1.state_dict().values()
        m_std = model2.state_dict().values()
        for i, (e, m) in enumerate(zip(e_std, m_std)):
            e.copy_(m)

    torch.save({'state_dict': model1.state_dict()}, 'weights/model.pt')


def check_args(kwargs):
    resample = kwargs.pop('resample', Image.BILINEAR)
    if isinstance(resample, (list, tuple)):
        kwargs['resample'] = random.choice(resample)
    else:
        kwargs['resample'] = resample


def shear_x(image, factor, **kwargs):
    factor = (factor / 10.) * 0.3
    factor = -factor if random.random() > 0.5 else factor

    check_args(kwargs)
    return image.transform(image.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)


def shear_y(image, factor, **kwargs):
    factor = (factor / 10.) * 0.3
    factor = -factor if random.random() > 0.5 else factor

    check_args(kwargs)
    return image.transform(image.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs)


def translate_x_rel(image, factor, **kwargs):
    factor = (factor / 10.) * 0.45
    factor = -factor if random.random() > 0.5 else factor
    pixels = factor * image.size[0]

    check_args(kwargs)
    return image.transform(image.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_rel(image, factor, **kwargs):
    factor = (factor / 10.) * 0.45
    factor = -factor if random.random() > 0.5 else factor
    pixels = factor * image.size[1]

    check_args(kwargs)
    return image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def translate_x_abs(image, factor, **kwargs):
    factor = (factor / 10.) * float(int(256 * 0.45))  # image size = 256
    factor = -factor if random.random() > 0.5 else factor

    check_args(kwargs)
    return image.transform(image.size, Image.AFFINE, (1, 0, factor, 0, 1, 0), **kwargs)


def translate_y_abs(image, factor, **kwargs):
    factor = (factor / 10.) * float(int(256 * 0.45))  # image size = 256
    factor = -factor if random.random() > 0.5 else factor

    check_args(kwargs)
    return image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, factor), **kwargs)


def rotate(image, factor, **kwargs):
    factor = (factor / 10.) * 30.
    factor = -factor if random.random() > 0.5 else factor

    check_args(kwargs)
    return image.rotate(factor, **kwargs)


def invert(image, _, **__):
    return ImageOps.invert(image)


def equalize(image, _, **__):
    return ImageOps.equalize(image)


def solar1(image, factor, **__):
    return ImageOps.solarize(image, int((factor / 10.) * 256))


def solar2(image, factor, **__):
    return ImageOps.solarize(image, 256 - int((factor / 10.) * 256))


def solar3(image, factor, **__):
    lut = []
    for i in range(256):
        if i < 128:
            lut.append(min(255, i + int((factor / 10.) * 110)))
        else:
            lut.append(i)
    if image.mode in ("L", "RGB"):
        if image.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return image.point(lut)
    else:
        return image


def poster1(image, factor, **__):
    factor = int((factor / 10.) * 4)
    if factor >= 8:
        return image
    return ImageOps.posterize(image, factor)


def poster2(image, factor, **__):
    factor = 4 - int((factor / 10.) * 4)
    if factor >= 8:
        return image
    return ImageOps.posterize(image, factor)


def poster3(image, factor, **__):
    factor = int((factor / 10.) * 4) + 4
    if factor >= 8:
        return image
    return ImageOps.posterize(image, factor)


def contrast1(image, factor, **__):
    factor = (factor / 10.) * .9
    factor = 1.0 + -factor if random.random() > 0.5 else factor
    return ImageEnhance.Contrast(image).enhance(factor)


def contrast2(image, factor, **__):
    return ImageEnhance.Contrast(image).enhance((factor / 10.) * 1.8 + 0.1)


def contrast3(image, _, **__):
    return ImageOps.autocontrast(image)


def color1(image, factor, **__):
    factor = (factor / 10.) * .9
    factor = 1.0 + -factor if random.random() > 0.5 else factor
    return ImageEnhance.Color(image).enhance(factor)


def color2(image, factor, **__):
    return ImageEnhance.Color(image).enhance((factor / 10.) * 1.8 + 0.1)


def brightness1(image, factor, **__):
    factor = (factor / 10.) * .9
    factor = 1.0 + -factor if random.random() > 0.5 else factor
    return ImageEnhance.Brightness(image).enhance(factor)


def brightness2(image, factor, **__):
    return ImageEnhance.Brightness(image).enhance((factor / 10.) * 1.8 + 0.1)


def sharpness1(image, factor, **__):
    factor = (factor / 10.) * .9
    factor = 1.0 + -factor if random.random() > 0.5 else factor

    return ImageEnhance.Sharpness(image).enhance(factor)


def sharpness2(image, factor, **__):
    return ImageEnhance.Sharpness(image).enhance((factor / 10.) * 1.8 + 0.1)


TRANSFORMS = [color1,
              color2,
              solar1,
              solar2,
              solar3,
              rotate,
              invert,
              poster1,
              poster2,
              poster3,
              shear_x,
              shear_y,
              equalize,
              contrast1,
              contrast2,
              contrast3,
              sharpness1,
              sharpness2,
              brightness1,
              brightness2,
              translate_x_rel,
              translate_y_rel,
              translate_x_abs,
              translate_y_abs]


class RandomAugment:
    def __init__(self):
        self.kwargs = dict(fillcolor=(124, 116, 104),
                           resample=(Image.BILINEAR, Image.BICUBIC))

    def __call__(self, image):
        if random.random() > 0.5:
            return image
        for transform in numpy.random.choice(TRANSFORMS, 2):
            magnitude = min(10., max(0., random.gauss(9, 0.5)))
            image = transform(image, magnitude, **self.kwargs)
        return image


class AverageMeter(object):
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n=1):
        self.num = self.num + n
        self.sum = self.sum + v * n
        self.avg = self.sum / self.num


class RandomResize:
    def __init__(self, size=256):
        self.size = size

        self.scale = (0.08, 1.0)
        self.ratio = (3. / 4., 4. / 3.)

    @staticmethod
    def params(img, scale, ratio):
        for _ in range(10):
            target_area = random.uniform(*scale) * img.size[0] * img.size[1]
            aspect_ratio = math.exp(random.uniform(*(math.log(ratio[0]), math.log(ratio[1]))))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        i, j, h, w = self.params(img, self.scale, self.ratio)
        resample = random.choice((Image.BILINEAR, Image.BICUBIC))
        return functional.resized_crop(img, i, j, h, w, [self.size, self.size], resample)
