import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import pandas as pd
from sklearn.model_selection import train_test_split

from code.data.preprocessing import preprocess
from code.data.preprocessing import bytecode_to_image

from code.data import mining


class BytecodeTokenDataset(data.Dataset):
    def __init__(self, df):
        df = preprocess(df)
        self.bytecodes = df['bytecode']
        self.tokens = df['tokens']

    def __getitem__(self, idx):
        return self.bytecodes[idx], self.tokens[idx]

    def __len__(self):
        return len(self.tokens)


class BytecodePonziDataset(data.Dataset):
    def __init__(self, df):
        result = pd.DataFrame(columns=['address', 'bytecode'])
        for i, row in df.iterrows():
            result['bytecode'].append(bytecode_to_image(row['bytecode']))
        df = result
        self.bytecodes = df['bytecode']
        self.is_ponzi = df['is_ponzi']

    def __getitem__(self, idx):
        return self.bytecodes[idx], self.is_ponzi

    def __len__(self):
        return len(self.bytecodes)


# BytecodeTokenDataset
# Read verified contracts
byt_token = pd.read_csv('../dataset/verified_contracts.csv')
# Split contracts into train/val/test
byt_token_trainval, byt_token_test = train_test_split(byt_token,
                                                      test_size=0.1,
                                                      random_state=42)
byt_token_train, byt_token_val = train_test_split(byt_token_trainval,
                                                  test_size=0.1,
                                                  random_state=42)

# BytecodePonziDataset
byt_ponzi_cols = ['Id',
                  'Name',
                  'address',
                  'R1',
                  'R2',
                  'R3',
                  'R4',
                  'is_ponzi',
                  'Owner',
                  'Type',
                  'Owner holds back some money',
                  'Unchecked Send',
                  'Owner functions to withdraw',
                  'Owner functions to change parameters'	
                  'Return fees if low',
                  'Similar contracts',
                  'Certified date',
                  'date',
                  'Notes']
# Read raw ponzi data
byt_ponzi = pd.read_csv('../dataset/Ponzi.csv', columns=byt_ponzi_cols)

# Mine bytecodes from BigQuery to add to raw ponzi data
bytecodes = pd.DataFrame()
for address in byt_ponzi.address:
    bytecodes.append(mining.get_sourcecode_from_address(address))
byt_ponzi['bytecode'] = bytecodes

# Select just address, ponzi_status, bytecode and save to new CSV
byt_token[['address', 'is_ponzi', 'bytecode']].to_csv('../dataset/cleaned_ponzi',
                                                      header=True,
                                                      index=False)
# Split Ponzi set to train/val/test
byt_ponzi_trainval, byt_ponzi_test = train_test_split(byt_ponzi,
                                                      test_size=0.05,
                                                      random_state=42)
byt_ponzi_train, byt_ponzi_val = train_test_split(byt_ponzi_trainval,
                                                  test_size=0.05,
                                                  random_state=42)
