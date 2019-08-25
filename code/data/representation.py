import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import pandas as pd

from code.data.preprocessing import preprocess
from code.data.preprocessing import bytecode_to_image


class BytecodeTokenDataset(data.Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        df = preprocess(df)
        self.bytecodes = df['bytecode']
        self.tokens = df['tokens']

    def __getitem__(self, idx):
        return self.bytecodes[idx], self.tokens[idx]

    def __len__(self):
        return len(self.tokens)


class BytecodePonziDataset(data.Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        result = pd.DataFrame(columns=['address', 'bytecode'])
        for i, row in df.iterrows():
            result['bytecode'].append(bytecode_to_image(row['bytecode']))
        df = result
        self.bytecodes = df['bytecode']

    def __getitem__(self, idx):
        return self.bytecodes[idx]

    def __len__(self):
        return len(self.bytecodes)
