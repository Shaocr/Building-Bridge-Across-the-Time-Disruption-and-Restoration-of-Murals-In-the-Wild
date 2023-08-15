# Building Bridge Across the Time: Disruption and Restoration of Murals In the Wild

---

## Environments

We run this code on an Ubuntu 20.04.5 server equipped with an Intel(R) Xeon(R) Gold 6230R CPU and 8 Nvidia(R) RTX 3090 GPUs. And the packages can be found in the file `requirements.txt`.

## Datasets
You can download the datasets from [here](https://drive.google.com/drive/folders/1DhHkSQIBAMtDwW3xcPhTIA3HoZGwfD1l)

- prepare the image data in `datasets/dunhuang/{}/`, e.g., `datasets/dunhuang/train_cond/1.png`
  
  - `train_cond`: damaged images (training set)
  
  - `train_ref`: ground truth (training set)
  
  - `test_cond`: damaged images (testing set)

  - `test_ref`: ground truth (testing set)


## Bash scripts

This repository can also reproduce the results via the bash script provided:

- run train script `./train.sh`

- run test script `./test.sh`
