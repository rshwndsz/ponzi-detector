import pandas as pd
import numpy as np
from nltk.metrics import edit_distance


def bytecode_to_image(bytecode):
    """
    Convert raw bytecode to a 128*128 image

    Can convert max of 128*128*3 bytes of bytecode
    rest is truncated
    :param bytecode: Raw bytecode as string
    :return: 128*128 normalized image ndarray
    """
    bytecode = bytecode[2:128*128+1]
    image = np.array([int(bytecode[i: i+3], 16)
                      for i in range(0, len(bytecode), 3)])
    image /= max(image)
    return image


def tokens_to_onehot(tokens):
    reference_tokens = ['transfer', 'burn',
                        'mint', 'withdraw',
                        'trade', 'airdrop',
                        'crowdsale', 'lock',
                        'pause', 'freeze',
                        'whitelist', 'remove',
                        'own', 'game',
                        'upgrade', 'destroy'
                        ]
    token_strings = list(tokens.keys())
    one_hot = np.zeros((16, 1))
    for i, ref_token in enumerate(reference_tokens):
        for dirty_token in tokens:
            if edit_distance(dirty_token, ref_token) <= 2:
                one_hot[i] = 1
                break
    return one_hot


def preprocess(df):
    result = pd.DataFrame(columns=['address', 'bytecode', 'tokens'])
    for i, row in df.iterrows():
        result['bytecode'].append(bytecode_to_image(row['bytecode']))
        result['tokens'].append(tokens_to_onehot(row['tokens']))
    return result
