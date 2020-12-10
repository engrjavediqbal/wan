# WAN: Weakly-Supervised Domain Adaptation for Built-up Region Segmentation in Aerial and Satellite Imagery

###  ISPRS Journal of Photogrametery and Remote Sensing

By Javed Iqbal and Mohsen Ali

### Update
- **2020.12.09**: code released for LT-WAN and OS-WAN

### Contents
0. [Introduction](#introduction)
0. [Requirements](#requirements)
0. [Setup](#models)
0. [Usage](#usage)
0. [Results](#results)
0. [Note](#note)
0. [Citation](#citation)

### Introduction
This repository contains the weakly supervised learning framwork for domain adaptation of built-up regions segmnentation based on the work described in ISPRS Photogrametery and Remote Sensing 2020 paper "[WAN: Weakly-Supervised Domain Adaptation for Built-up Region Segmentation in Aerial and Satellite Imagery]". 
(https://arxiv.org/pdf/2007.02277.pdf).

### Requirements:
The code is tested in Ubuntu 16.04. It is implemented based on Keras with tensorflow backend and Python 3.5. For GPU usage, the maximum GPU memory consumption is about 7 GB in a single GTX 1080.


### Setup
We assume you are working in wan-master folder.

0. Datasets:
- Download [Rwanda](https://drive.google.com/file/d/1RDrzeUUzSJR4YTG5hlegVWCAwMUEkRMF/view?usp=sharing) dataset. 
- Put downloaded data in "datasets" folder.

### Usage
0. Set the PYTHONPATH environment variable:
~~~~
cd wan-master

~~~~
1. Adaptation
- OSA: Output space Adaptation:
~~~~

python adapt_OSA.py --data-dir path_to_dataset_folder --data-list-train training_images_list --data-list-val validation_images_list
~~~~

- LTA: Output space Adaptation:
~~~~
python adapt_LTA.py --data-dir path_to_dataset_folder --data-list-train training_images_list --data-list-val validation_images_list
~~~~
3. 
- To run the code, you need to set the data paths of source data (data-root) and target data (data-root-tgt) by yourself. Besides that, you can keep other argument setting as default.

4. Evaluation

~~~~
~~~~

5. Train in source domain
~~~~
python train.py --data-dir path_to_dataset_folder --data-list-train training_images_list --data-list-val validation_images_list
~~~~





### Citation:
If you found this useful, please cite our [paper](https://www.sciencedirect.com/science/article/pii/S0924271620301829). 

>@inproceedings{iqbal2020weakly,  
>&nbsp; &nbsp; &nbsp;    title={Weakly-supervised domain adaptation for built-up region segmentation in aerial and satellite imagery},  
>&nbsp; &nbsp; &nbsp;     author={Iqbal, Javed and Ali, Mohsen},  
>&nbsp; &nbsp; &nbsp;     journal={ISPRS Journal of Photogrammetry and Remote Sensing}, 
>&nbsp; &nbsp; &nbsp;     volume={167},
>&nbsp; &nbsp; &nbsp;     pages={263--275},
>&nbsp; &nbsp; &nbsp;     year={2020},
>&nbsp; &nbsp; &nbsp;     publisher={Elsevier}
>}


Contact: javed.iqbal@itu.edu.pk
