# DHDRNet

This work accompanies my thesis: "**Learned Exposure Selection for High Dynamic Range Image Synthesis**".

DHDRNet, or **D**ual-photo **HDR** **Net**work lets you effectively create HDR images with 60% fewer resources when compared with standard [Exposure Fusion](https://en.wikipedia.org/wiki/Exposure_fusion) [1].

It introduces a convolutional network that predicts the two optimal photos needed to generate a high quality HDR image with Exposure Fusion. It also introduces a data generation system that creates synthetically exposed images and accounts for metadata corruption in order to generate a high quality training set.

## Usage

There are two main components of this system. Firstly, the data generation system is used to take a set of HDR DNG files (like those found in the HDR+ [2] dataset from Google) and generate synthetic exposures, statistics, and exposure-fused images for training purposes.

The image reconstructions are generated in advance of training, instead of on-demand as there are computational constraints involved in training-time generation of Exposure Fusion images (the Exposure Fusion algorithm is CPU-bound).

This project is created with [Poetry](https://python-poetry.org), and as such it is recommended to create the virtual environment with:

```sh
git clone https://github.com/smsegal/DHDRNet.git
cd DHDRNet
poetry install
```

### Data Generation

Download the HDR+ DNG files from google with:
```sh
python -m dhdrnet.data_prep download --out=./foo
```
This utilizes the [Google Cloud SDK](https://cloud.google.com/sdk/docs/quickstart), so make sure you've signed in and authenticated. 

After the DNG files are downloaded, the synthetic exposures and fused images can be generated with: 

```sh
python -m dhdrnet.data_prep --download-dir=./foo --out=./bar
```

See the `python -m dhdrnet.data_prep --help` output for further command line arguments that can customize aspects of the data generation process.

The generated data will be stored in the specified directories for use in later training. 

### Training

Several network backbones were explored and tested for efficacy and efficiency. The network that struck the best balance of performance to model size was MobileNetV2, but other architectures are included for comparison. The other network backbones used here are ResNet, SqueezeNet, MobileNetV1, and EfficientNet. See my thesis for more details.

To train a network from scratch:

```sh
python -m dhdrnet.train --model=mobilenetv2 --data-dir=./bar --logdir=./baz
```

This will by default output tensorboard logs in `logs/`. 

See the output of `python -m dhdrnet.train --help` for more options that can customize training.



Code written with PyTorch and PyTorch Lightning.

[1] T. Mertens, J. Kautz, and F. Van Reeth, “Exposure Fusion: A Simple and Practical Alternative to High Dynamic Range Photography,” Computer Graphics Forum, vol. 28, no. 1, pp. 161–171, Mar. 2009, doi: 10.1111/j.1467-8659.2008.01171.x.

[2] S. W. Hasinoff et al., “Burst photography for high dynamic range and low-light imaging on mobile cameras,” ACM Trans. Graph., vol. 35, no. 6, pp. 1–12, Nov. 2016, doi: 10.1145/2980179.2980254.
