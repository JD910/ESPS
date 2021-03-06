# EfficientNetV2-based survival benefit prediction system (ESBP)

[![standard-readme compliant](https://img.shields.io/badge/Readme-standard-brightgreen.svg?style=flat-square)](https://github.com/JD910/ESLN/blob/main/README.md)
![](https://img.shields.io/badge/Pytorch-1.7.1-brightgreen.svg?style=flat-square)

An EfficientNetV2-based survival benefit prediction system was developed to predict the additional survival benefit of EGFR-TKIs and ICIs in stage IV NSCLC patients.

*main.py:* Includes the entry function of ESBP and the code for training and validation.  

  > input_dir_train: The path where your training dataset is stored.
  > 
  > input_dir_val: The path where your validation dataset is stored.
  > 
  > input_dir_test: The path where your test dataset is stored.
  > 
  > * For the reproducibility, a subdataset (anonymized) is made publicly accessible in this repository. Readers can directly use this dataset to run the source code of the ESBP in this study.

*HDF5_read.py:* Defines the function of  CT image reading.

  > train_data: The input for ESBP. Examples of the input images are presented in Fig. S1 below.
  > 
  > target_data: The label for training and test.
  > 
  > keys[index]: The name of each image. All data are anonymized, and *P1_1* represents the first image of the first patient. 

*Weight folder:* Download the "best_pt_OA.pt.tar.gz*" files and use the following command to extract the well-trained ESPS model.

  > cat best_pt_OA.pt.tar.gz* | tar -xzv
  > 

*nets folder:* The definition of the ESBP network.

*utils folder:* Necessary functions used in the ESBP.

*Test folder:* The publicly accessible test data for reproducibility. Due to the size limitation, use the following command to extract the data.

  > cat Open_Access_Data.hdf5.tar.gz* | tar -xzv
  > 
 
<div align=left><img width="1000" height="123" src="https://github.com/JD910/ESPS/blob/main/utils/Examples.jpg"/></div><br />

**Fig. S1. Examples of the input images of ESBP. a(1) to a(4) and a(5) to a(8) represent two different patients.**<br />
