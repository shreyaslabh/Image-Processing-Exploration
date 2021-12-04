from skimage import io
import matplotlib.pyplot as plt

from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.filters import threshold_otsu

import numpy as np

import glob

from scipy.stats import linregress

path = 'dataset/*.*'

time_dim = []
scratch_area_dim = []

time = 0
for image in glob.glob(path):
    
    img = io.imread(image)
    
    # Thresholding wrt Entropy
    entropy_img = entropy(img, disk(10))
    
    threshold = threshold_otsu(entropy_img)
    
    # Binary Segmentation
    binary_img = entropy_img <= threshold
    
    scratch_area = np.sum(binary_img == True)
    
    scratch_area_dim.append(scratch_area)
    time_dim.append(time)
    
    time += 1


for t,a in zip(time_dim, scratch_area_dim):
    print(f"Area at Time {t} = {a} pixels")
    
plt.plot(time_dim, scratch_area_dim, 'bo')

slope, intercept, r, p, err = linregress(time_dim, scratch_area_dim)

print(f"Equation of Line: y = {slope} * x + {intercept}")
print(f"R\N{SUPERSCRIPT TWO} = {r**2}")

