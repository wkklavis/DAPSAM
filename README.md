# Prompting Segment Anything Model with Domain-Adaptive Prototype for Generalizable Medical Image Segmentation (DAPSAM)

This is the official code of our MICCAI 2024 paper [DAPSAM](https://baidu.com) 🥳

<div align=center>
	<img src="figures/pipeline.png" width=75%/>
</div>

## Requirement
``pip install -r requirements.txt``


## Data Preparation
[Prostate Segmentation](https://liuquande.github.io/SAML/)

[RIGA+ Segmentation](https://zenodo.org/records/6325549)

Please download the pretrained [SAM model](https://drive.google.com/file/d/1_oCdoEEu3mNhRfFxeWyRerOKt8OEUvcg/view?usp=share_link) 
(provided by the original repository of SAM) and put it in the ./pretrained folder. 

What's more, we also provide well-trained models at [Release](https://github.com/wkklavis/DAPSAM/releases/tag/v1.0.0). Please put it in the ./snapshot folder for evaluation. 


## Prostate Segmentation

We take the setting using RUNMC (source domain) and other five datasets (target domains) as the example.

```
cd prostate
# Training
CUDA_VISIBLE_DEVICES=0 python train.py --root_path dataset_path --output output_path --Source_Dataset RUNMC --Target_Dataset BIDMC BMC HK I2CVB UCL
# Test
CUDA_VISIBLE_DEVICES=0 python test.py --root_path dataset_path --output_dir output_path --Source_Dataset RUNMC --Target_Dataset BIDMC BMC HK I2CVB UCL --snapshot snapshot_path
```


## RIGA+ Segmentation

We take the setting using BinRushed (source domain) and other three datasets (target domains) as the example.

```
cd fundus
# Training
CUDA_VISIBLE_DEVICES=0 python train.py --root_path dataset_path --output output_path --Source_Dataset BinRushed --Target_Dataset MESSIDOR_Base1 MESSIDOR_Base2 MESSIDOR_Base3
# Test
CUDA_VISIBLE_DEVICES=0 python test.py --root_path dataset_path --output output_path --Source_Dataset BinRushed --Target_Dataset MESSIDOR_Base1 MESSIDOR_Base2 MESSIDOR_Base3 --snapshot snapshot_path
```


## Cite 
If you find this code useful, please cite
~~~

~~~

## Acknowledgement

We appreciate the developers of [Segment Anything Model](https://github.com/facebookresearch/segment-anything). 
The code of DAPSAM is built upon [SAMed](https://github.com/hitachinsk/SAMed), and we express our gratitude to these projects.