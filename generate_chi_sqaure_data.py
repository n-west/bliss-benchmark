

import numpy as np
from scipy.stats import skew

import h5py
%matplotlib widget
import matplotlib.pyplot as plt

num_intergrations = 51


base = h5py.File("/datag/public/voyager_2020/single_coarse_channel/single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5", mode="r")


data_attrs = {}
for k in base["data"].attrs.keys():
    print(f"{k}    :::  {base['data'].attrs[k]}")
    data_attrs[k] = base['data'].attrs[k]


data_attrs["source_name"] = "synthetic noise"
data_attrs["tsamp"] = abs(1/(data_attrs["foff"]*1e6)) * num_intergrations


np.random.seed(42)

# Base of dual polarization is 4 DOF, each integrated spectrum accumulates 4 DOF at a time
r = np.float32(1*np.random.randn(16,4*num_intergrations,2**20) + 0.0)

# putting the square in chi-squared
r = (r*r).sum(1)

r = r.reshape((r.shape[0], 1, r.shape[-1]))


print(np.std(r, dtype=np.float32))
print(np.mean(r, dtype=np.float32))

print(skew(r.flatten()))


f = h5py.File(f"just_noise_chisq_{4*num_intergrations}dof.h5", mode="w")
f["data"] = r
for k,v in data_attrs.items():
    f["data"].attrs[k] = v
for k in base.attrs.keys():
    f.attrs[k] = base.attrs[k]
f.close()
