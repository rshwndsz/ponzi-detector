from torch.utils import data
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

from code.data.preprocessing import preprocess
from code.data.preprocessing import bytecode_to_image
from code.data import mining


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


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
        result = pd.DataFrame(columns=['address', 'bytecode', 'is_ponzi'])
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
# DataLoaders
byt_token_trainloader = data.DataLoader(byt_token_train,
                                        batch_size=2,
                                        shuffle=True,)
byt_token_valloader = data.DataLoader(byt_token_val,
                                      batch_size=2,
                                      shuffle=True)
byt_token_testloader = data.DataLoader(byt_token_test,
                                       batch_size=2,
                                       shuffle=False)

# BytecodePonziDataset
# Read raw ponzi data
byt_ponzi = pd.read_csv('../dataset/Ponzi.csv')
# Split Ponzi set to train/val/test
byt_ponzi_trainval, byt_ponzi_test = train_test_split(byt_ponzi,
                                                      test_size=0.05,
                                                      random_state=42)
byt_ponzi_train, byt_ponzi_val = train_test_split(byt_ponzi_trainval,
                                                  test_size=0.05,
                                                  random_state=42)
# DataLoaders
byt_ponzi_trainloader = data.DataLoader(byt_ponzi_train,
                                        batch_size=2,
                                        shuffle=True,)
byt_ponzi_valloader = data.DataLoader(byt_ponzi_val,
                                      batch_size=2,
                                      shuffle=True)
byt_ponzi_testloader = data.DataLoader(byt_ponzi_test,
                                       batch_size=2,
                                       shuffle=False)

if __name__ == '__main__':
    import os
    if not os.path.exists('../dataset/cleaned_ponzi.csv'):
        logger.warning('Ponzi set has been cleaned and bytecode added.')
    else:
        # BytecodePonziDataset
        byt_ponzi_cols = ['Id',
                          'Id2',
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
        byt_ponzi = pd.read_csv('../dataset/Ponzi.csv', names=byt_ponzi_cols)
        logger.debug(byt_ponzi.head())

        # Mine bytecodes from BigQuery to add to raw ponzi data
        logger.info('Adding bytecodes to ponzi data...')
        bytecodes = []
        query_job = mining.get_bytecode_from_addresses(set(byt_ponzi['address']))

        for row in query_job:
            bytecodes.append(row[0])

        byt_ponzi['bytecode'] = pd.DataFrame(bytecodes)

        byt_ponzi['is_ponzi'] = pd.DataFrame([1]*len(byt_ponzi))

        # Add not ponzi data
        extra_not_ponzi = pd.read_csv('../dataset/verified_contracts.csv')
        df = extra_not_ponzi.sample(n=200)
        df['is_ponzi'] = pd.DataFrame([0]*len(df))
        byt_ponzi = pd.concat([byt_ponzi[['address', 'bytecode', 'is_ponzi']],
                              df[['address', 'bytecode', 'is_ponzi']]])

        # Select just address, ponzi_status, bytecode and save to new CSV
        byt_ponzi[['address', 'bytecode', 'is_ponzi']].to_csv('../dataset/cleaned_ponzi.csv',
                                                              header=True,
                                                              index=False)
        logger.info('Ponzi set cleaned!')
