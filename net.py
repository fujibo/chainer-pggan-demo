import chainer
from chainer import functions as F
from chainer import links as L


class Generator(chainer.Chain):
    def __init__(self, in_size=512, out_channel=3):
        super(Generator, self).__init__()
        with self.init_scope():
            self.fc7_1 = L.Linear(in_size, 4*4*512, nobias=True)
            self.b7_1 = L.Bias(shape=(512, ))
            self.conv7_2 = L.Convolution2D(512, 512, ksize=3, pad=1)
            
            self.conv6_1 = L.Convolution2D(512, 512, ksize=3, pad=1)
            self.conv6_2 = L.Convolution2D(512, 512, ksize=3, pad=1)

            self.conv5_1 = L.Convolution2D(512, 512, ksize=3, pad=1)
            self.conv5_2 = L.Convolution2D(512, 512, ksize=3, pad=1)

            self.conv4_1 = L.Convolution2D(512, 512, ksize=3, pad=1)
            self.conv4_2 = L.Convolution2D(512, 512, ksize=3, pad=1)

            self.conv3_1 = L.Convolution2D(512, 256, ksize=3, pad=1)
            self.conv3_2 = L.Convolution2D(256, 256, ksize=3, pad=1)

            self.conv2_1 = L.Convolution2D(256, 128, ksize=3, pad=1)
            self.conv2_2 = L.Convolution2D(128, 128, ksize=3, pad=1)

            self.conv1_1 = L.Convolution2D(128, 64, ksize=3, pad=1)
            self.conv1_2 = L.Convolution2D(64, 64, ksize=3, pad=1)

            self.conv0_0 = L.Convolution2D(64, out_channel, ksize=1)

    
    def __call__(self, x, with_dict=False):
        output_dict = dict()
        output_dict['input'] = x

        # 512 -> 4x4
        x = _pixel_norm(chainer.Variable(x), eps=1e-8)
        x = F.reshape(self.fc7_1(x), (-1, 512, 4, 4))
        x = _pixel_norm(F.leaky_relu(self.b7_1(x)), eps=1e-8)
        x = _pixel_norm(F.leaky_relu(self.conv7_2(x)), eps=1e-8)
        output_dict['x2'] = x

        # 4x4 -> 8x8
        x = F.unpooling_2d(x, ksize=2, cover_all=False)
        x = _pixel_norm(F.leaky_relu(self.conv6_1(x)), eps=1e-8)
        x = _pixel_norm(F.leaky_relu(self.conv6_2(x)), eps=1e-8)
        output_dict['x3'] = x

        # 8x8 -> 16x16
        x = F.unpooling_2d(x, ksize=2, cover_all=False)
        x = _pixel_norm(F.leaky_relu(self.conv5_1(x)), eps=1e-8)
        x = _pixel_norm(F.leaky_relu(self.conv5_2(x)), eps=1e-8)

        # 16x16 -> 32x32
        x = F.unpooling_2d(x, ksize=2, cover_all=False)
        x = _pixel_norm(F.leaky_relu(self.conv4_1(x)), eps=1e-8)
        x = _pixel_norm(F.leaky_relu(self.conv4_2(x)), eps=1e-8)

        # 32x32 -> 64x64
        x = F.unpooling_2d(x, ksize=2, cover_all=False)
        x = _pixel_norm(F.leaky_relu(self.conv3_1(x)), eps=1e-8)
        x = _pixel_norm(F.leaky_relu(self.conv3_2(x)), eps=1e-8)

        # 256x64x64 -> 128x128x128
        x = F.unpooling_2d(x, ksize=2, cover_all=False)
        x = _pixel_norm(F.leaky_relu(self.conv2_1(x)), eps=1e-8)
        x = _pixel_norm(F.leaky_relu(self.conv2_2(x)), eps=1e-8)

        # 128x128x128 -> 64x256x256
        x = F.unpooling_2d(x, ksize=2, cover_all=False)
        x = _pixel_norm(F.leaky_relu(self.conv1_1(x)), eps=1e-8)
        x = _pixel_norm(F.leaky_relu(self.conv1_2(x)), eps=1e-8)

        # 64x256x256 -> 1x256x256 (ToRGB_lod0)
        img = self.conv0_0(x)
        images_out = img
        return images_out


def _pixel_norm(x, eps=1e-8):
    alpha = 1.0 / F.sqrt(F.mean(F.square(x), axis=1, keepdims=True) + eps)
    return F.broadcast_to(alpha, x.array.shape) * x
