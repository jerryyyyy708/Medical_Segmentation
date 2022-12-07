
# Medical Segmentation
交大資訊工程專題(二) 進度紀錄


## Dataset

#### Kvasir-SEG 

<img src="https://i.imgur.com/aWIZQ1W.png" width="600px">

https://datasets.simula.no/kvasir-seg/


## Model

#### UNet

<img src="https://i.imgur.com/wCxIRcE.png" width="600px">


## Files:
	1. image_resizer.py: Resize the images into 256x256 (so that my GPU can afford).
	2. medical_segmentation.ipynb: Main code to run everything.
	3. models.py: Class of Model and Dataset.
	4. train.py: Functions of train, valid, test and save result. 




## Results

(1) Original Image (2) Ground Truth (3) Result

<img src="https://i.imgur.com/9pzI2fg.png" width="450px">


## Reference

UNet https://arxiv.org/pdf/1505.04597.pdf

