import os
from PIL import Image
import random

def main():
    src = 'Dataset/train_images/'
    mask = 'Dataset/train_masks/'

    dest = 'Dataset/paste_images/'
    mdest = 'Dataset/paste_masks/'

    picts = os.listdir(src)
    for i in picts:
        if os.path.isfile(os.path.join(dest,'paste_'+i)):
            print('hi')
            continue
        input('press enter to show image')
        index = random.randint(0,899)
        im = Image.open(os.path.join(src,i))
        mk = Image.open(os.path.join(mask,i))

        rndim = Image.open(os.path.join(src,picts[index]))
        rnmk = Image.open(os.path.join(mask,picts[index]))

        im.show()
        x1,y1,x2,y2= input('Enter edges of img to cut\n').split()

        x3,y3 = str(random.randint(0,200)),str(random.randint(0,200))

        polyp = im.crop((int(x1),int(y1),int(x2),int(y2)))
        gt = mk.crop((int(x1),int(y1),int(x2),int(y2)))

        if int(x3) + int(x2) - int(x1) >= 256:
            x3 = str(int(x3) - (int(x3) + int(x2) - int(x1) - 256))
        if int(y3) + int(y2) - int(y1) >= 256:
            y3 = str(int(y3) - (int(y3) + int(y2) - int(y1) - 256))
        
        rndim.paste(polyp,(int(x3),int(y3)))
        rnmk.paste(gt,(int(x3),int(y3)))

        rndim.save(os.path.join(dest,'paste_'+i))
        rnmk.save(os.path.join(mdest,'paste_'+i))

main()