#show how big is the polyp
import os
from PIL import Image
import numpy as np
a = []
root = 'Dataset/train_masks'
for i in os.listdir(root):
    img = os.path.join(root, i)
    mask = np.array(Image.open(img).convert('L'))
    mask[mask!=0] = 1
    a.append(np.sum(mask))
    #print(i,np.sum(np.array(mask)))
a = sorted(a)
print(a[int(len(a)/3)],a[int(len(a)/3*2)],a[len(a)-1])