# DHDRNet

This work accompanies my thesis: "Exposure Fusion with Learned Image Selection". 

It introduces a convolutional network that predicts the two optimal photos needed to generate a high quality HDR image with Exposure Fusion. It also introduces a data generation system that creates synthetically exposed images and accounts for metadata corruption in order to generate a high quality training set. 

This system effectively cuts the requirements of common Exposure Fusion HDR systems from needing 5 source images to only two, in an intelligent manner. 

Code written with PyTorch. 

This code hasn't necessarily been cleaned up for public view. Lots of in progress research notes and experimentation contained in the notebooks and maybe the code comments :)