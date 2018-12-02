import chainer
from chainer.backends import cuda
from chainercv import utils

import numpy as np

from net import Generator


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./progressive_growing_of_gans/Gs_chainer.npz')
    args = parser.parse_args()

    chainer.config.train = False

    latent = np.random.randn(4, 512).astype(np.float32)

    generator = Generator()
    chainer.serializers.load_npz(args.model_path, generator)

    with chainer.no_backprop_mode():
        img = generator(latent)
        print(img.shape)

    # [-1, 1] -> [0, 255]
    image = cuda.to_cpu(img.array) * 127.5 + 127.5
    image = image.clip(0.0, 255.0).astype(np.float32)
    utils.write_image(utils.tile_images(image, 2), 'out.png')


if __name__ == "__main__":
    main()
