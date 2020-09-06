import hdf5storage
import h5py
import numpy as np
import pickle as pkl

mat = h5py.File('nyu_depth_v2_labeled.mat','r')
print(mat.keys())
df_x = mat['images']
images = []
df_y = mat['depths']
depths = []

for i in df_x:
    img = np.asarray(i)
    img = np.transpose(img,axes=[2,1,0])
    images.append(img)

for j in df_y:
    depth = np.asarray(j)
    depth = np.transpose(depth,axes=[1,0])
    depth = depth.reshape((depth.shape[0],depth.shape[1],1))
    depth = (depth / depth.max()) * 255
    print(depth.shape)
    depths.append(depth)



hf = h5py.File('dataset.h5','w')
hf.create_dataset('images',data=images)
hf.create_dataset('depths',data=depths)
hf.close()