import argparse
import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
from tqdm import tqdm
import random

def parse_args():
    parser = argparse.ArgumentParser(prog="image_resizer.py")
    parser.add_argument("--image", "-i", default='C:/Users/jerry/medical_segmentation/Dataset/train_images', type=str, required=False, help="Image Folder")
    parser.add_argument("--mask", "-m", default='C:/Users/jerry/medical_segmentation/Dataset/train_masks', type=str, required=False, help="Mask Folder")
    parser.add_argument("--outimg", "-o", default="C:/Users/jerry/medical_segmentation/Dataset/t1v2/", type=str, required=False, help="Output Folder")
    parser.add_argument("--outmask", "-k", default="C:/Users/jerry/medical_segmentation/Dataset/t2v2/", type=str, required=False, help="Output Folder")
    return parser.parse_args()

def main():
    args = parse_args()
    image_folder=os.listdir(args.image)
    mask_folder=os.listdir(args.mask)
    for i,(image , mask) in tqdm(enumerate(zip(image_folder, mask_folder))):
        assert image == mask
        ipath = os.path.join(args.image,image)
        mpath = os.path.join(args.mask,mask)
        img_file = Image.open(ipath).convert("RGB")
        mask_file = Image.open(mpath).convert("L")

        hflip = float(random.uniform(0,1)>0.5)
        vflip = float(random.uniform(0,1)>0.5)
        pers = float(random.uniform(0,1)>0.5)
        rotate = random.uniform(0,90)
        ccw = random.randint(192,320)#centercrop width
        cch = random.randint(192,320)#centercrop height
        perscale = random.uniform(0.25,0.75)

        img_file = F.rotate(img_file,rotate)
        mask_file = F.rotate(mask_file,rotate)

        transform = transforms.Compose([
            #transforms.RandomErasing(p=0.25,scale = (0.02,0.1),value=0,inplace=False),
            #transforms.RandomPerspective(distortion_scale= perscale,p = pers),
            transforms.CenterCrop((ccw,cch)), 
            transforms.RandomHorizontalFlip(p=hflip),
            transforms.RandomVerticalFlip(p=vflip),
        ])

        img_file = transform(img_file)
        mask_file = transform(mask_file)
        img_file = img_file.resize((256,256))
        mask_file = mask_file.resize((256,256))

        if random.uniform(0,1)>0.5:
            colortrans = transforms.Compose([
                transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1)
            ])
            img_file = colortrans(img_file)
        
        if random.uniform(0,1)>0.6:
            blur = transforms.GaussianBlur(kernel_size=(21, 21), sigma=(0.1, 10))
            img_file =blur(img_file)
            

        img_file.save(args.outimg+image)
        mask_file.save(args.outmask+image)
#class torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
        
main()