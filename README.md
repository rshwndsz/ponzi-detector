# Detecting fraud Ethereum smart contracts using Machine Learning

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

Download the dataset from [this link](https://) into `dataset/`.

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
