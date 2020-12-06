# DHDRNet

This work accompanies my thesis: "**Exposure Fusion with Learned Image Selection**".

DHDRNet, or **D**ual-photo **HDR** **Net**work lets you effectively create HDR images from only two captures, instead of the usual five commonly required.

It introduces a convolutional network that predicts the two optimal photos needed to generate a high quality HDR image with Exposure Fusion. It also introduces a data generation system that creates synthetically exposed images and accounts for metadata corruption in order to generate a high quality training set.

This system effectively cuts the requirements of common Exposure Fusion HDR systems from needing 5 source images to only two, in an intelligent manner.

Code written with PyTorch and PyTorch Lightning.

This code hasn't necessarily been cleaned up for public view. Lots of in progress research notes and experimentation contained in the notebooks and maybe the code comments :)
