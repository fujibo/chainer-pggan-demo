import numpy as np


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', default='./progressive_growing_of_gans/Gs.npz')
    parser.add_argument('--dest_path', default='./progressive_growing_of_gans/Gs_chainer.npz')
    args = parser.parse_args()

    model = np.load(args.src_path)

    model_dest = dict()
    dic = {'4x4': 7, '8x8': 6, '16x16': 5, '32x32': 4, '64x64': 3, '128x128': 2, '256x256': 1}

    for key, value in model.items():
        if key == 'lod':
            # value = 0.0
            continue
        
        key = key.replace('weight', 'W')
        key = key.replace('bias', 'b')
        head = key.split('/')[0]

        if head == '4x4':
            if key.startswith('4x4/Dense'):
                if key == '4x4/Dense/W':
                    key = key.replace('4x4/Dense', 'fc{}_1'.format(dic[head]))
                
                elif key == '4x4/Dense/b':
                    key = key.replace('4x4/Dense', 'b{}_1'.format(dic[head]))

            else:
                key = key.replace('4x4/Conv', 'conv{}_2'.format(dic[head]))

        elif 'x' in head:
            # when you use deconv
            if 'Conv0_up' in key:
                key = key.replace(f'{head}/Conv0_up',f'conv{dic[head]}_1')
                deconv_used = True
            elif 'Conv0' in key:
                key = key.replace(f'{head}/Conv0',f'conv{dic[head]}_1')
                deconv_used = False
            elif 'Conv1' in key:
                key = key.replace(f'{head}/Conv1',f'conv{dic[head]}_2')
            else:
                print(key, 'is not expected.')
            
        elif key.startswith('ToRGB'):
            if key.startswith('ToRGB_lod0'):
                key = key.replace('ToRGB_lod0', 'conv0_0')
            
            else:
                continue
        
        else:
            continue

        gain = np.sqrt(2)

        # dense W
        if len(value.shape) == 2:
            fan_in = value.shape[0]
            # gain=np.sqrt(2)/4 in Generator dense
            std = (gain / 4) / np.sqrt(fan_in)
            value = value.T * std

        # conv W
        elif len(value.shape) == 4:
            # (1, 1, 128, 1)
            if key == 'conv0_0/W':
                fan_in = value.shape[0] * value.shape[1] * value.shape[2]
                std = 1.0 / np.sqrt(fan_in)
            # (3, 3, 128, 256)
            else:
                fan_in = value.shape[0] * value.shape[1] * value.shape[3]
                std = gain / np.sqrt(fan_in)
                
            value = value.transpose(3, 2, 0, 1) * std
            
            if key.startswith('conv') and key.endswith('_1/W') and deconv_used:
                value_pad = np.zeros(value.shape[:2] + (value.shape[2]+2, value.shape[3]+2), dtype=np.float32)
                value_pad[:, :, 1:-1, 1:-1] = value
                value = value_pad[:, :, 1:, 1:] + value_pad[:, :, :-1, 1:] + value_pad[:, :, 1:, :-1] + value_pad[:, :, :-1, :-1]

        model_dest[key] = value

    np.savez(args.dest_path, **model_dest)


if __name__ == "__main__":
    main()
