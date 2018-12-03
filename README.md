# chainer-pggan-demo
This repository contains the **demo** code for the paper Progressive Growing of GANs implemented with Chainer.

## Requirements
- python3
- googledrivedownloader==0.3
- chainer==5.0.0
- chainercv==0.11.0
- environment for progressive_growing_of_gans

## Usage
1. Clone this repository.
```bash
$ git clone https://github.com/fujibo/chainer-pggan-demo.git --recursive
```

2. Extract weights of a tensorflow model in a npz format.
```bash
$ cd progressive_growing_of_gans/
$ python download_model.py
$ python covert_model.py
```

3. Adjust layer names and weights for Chainer.
```bash
$ cd ../
$ python convert_npz.py
```

4. Demonstrate.
```bash
$ python demo.py
```
