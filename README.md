# depthEstimation

estimation of depth from a single image.
dataset -> nyu depth dataset labeled dataset -> http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
nyu_dataset.py -> contains the code to format the data and save the images, depths values to a h5 file
depthEstimationCnnResnet -> implementation of the CNN + ResNet architecture from the paper, http://cs231n.stanford.edu/reports/2017/pdfs/203.pdf
depthEstimationAttn -> implementation of the architecture from the paper, https://arxiv.org/abs/1807.03959
