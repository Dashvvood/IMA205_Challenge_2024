# calculate mean and std deviation
from pathlib import Path
import cv2
import numpy as np
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()

parser.add_argument("--folder", default="./data")
opts, _ = parser.parse_known_args()

imageFilesDir = Path(opts.folder)
files = list(imageFilesDir.rglob('*.jpg'))

# Since the std can't be calculated by simply finding it for each image and averaging like  
# the mean can be, to get the std we first calculate the overall mean in a first run then  
# run it again to get the std.

mean = np.array([0.,0.,0.])
stdTemp = np.array([0.,0.,0.])
std = np.array([0.,0.,0.])

numSamples = len(files)

for i in tqdm(range(numSamples)):
    im = cv2.imread(str(files[i]))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(float) / 255.
    
    for j in range(3):
        mean[j] += np.mean(im[:,:,j])

mean = (mean/numSamples)

for i in tqdm(range(numSamples)):
    im = cv2.imread(str(files[i]))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(float) / 255.
    for j in range(3):
        stdTemp[j] += ((im[:,:,j] - mean[j])**2).sum()/(im.shape[0]*im.shape[1])

std = np.sqrt(stdTemp/numSamples)

print(mean)
print(std)

# out:
#[0.50707516 0.48654887 0.44091784]
#[0.26733429 0.25643846 0.27615047]