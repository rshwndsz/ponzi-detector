# Detecting fraud Ethereum smart contracts using Machine Learning

This project is the result of a 48hr sprint to create a blockchain-based deep learning project.

## Setting Up

Create a conda environment using the conda config file `environment.yml`.
conda config files work only on the OS they are generated on (Linux64 for Me).

```bash
conda create -f environment.yml
```

Install all dependencies using `conda install`.

```bash
cd name-of-model
```

Download the dataset from [this link](https://https://drive.google.com/open?id=1sSb0i7lTPpmo2UAKEVfp1ZwCJYDl1wj5) into `dataset/`.

## Training

Run the project using

```bash
python main.py --phase train
```

## Testing

Test the project using

```bash
python main.py --phase test
```

Note:  
Not all files may be up-to-date.
