import argparse
import os
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(prog="image_resizer.py")
    parser.add_argument("--input", "-i", default="", type=str, required=True, help="Input Folder")
    parser.add_argument("--output", "-o", default="", type=str, required=True, help="Output Folder")
    return parser.parse_args()

def main():
    args = parse_args()
    input_folder=os.listdir(args.input)
    for fname in input_folder:
        fpath=os.path.join(args.input, fname)
        if os.path.isdir(fpath):
            ofolder = os.path.join(args.output, fname)
            if not os.path.exists(ofolder):
                os.makedirs(ofolder)
            for img in os.listdir(fpath):
                ipath=os.path.join(fpath,img)
                if "images" in ipath:
                    pict = Image.open(ipath).convert("RGB")
                    pict = pict.resize((256,256))
                    opath = os.path.join(ofolder,img)
                    pict.save(opath)
                elif "masks" in ipath:
                    pict = Image.open(ipath).convert("L")
                    pict = pict.resize((256,256))
                    opath = os.path.join(ofolder,img)
                    pict.save(opath)

main()