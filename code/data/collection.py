import pandas as pd
from collections import Counter
import os
import logging

from code.data import mining
from code.data import cleaning
from code.config import config as cfg


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def collect_verified_contracts():
    # Paths to BigQuery results split into multiple files (100000 rows each)
    dataset_paths = [
        'bq-results-20190825-131705-h0cn745b0vnb.csv',
        'bq-results-20190825-132153-sloth1dfc1lz.csv',
        'bq-results-20190825-134911-yajahtsu6qok.csv'
    ]
    dataset_paths = [os.path.join(cfg.dataset_root, dataset_path)
                     for dataset_path in dataset_paths]

    # Concatenate BigQuery results (address, bytecode) into 1 CSV file
    logger.debug('Concatenating BigQuery results...')
    ethereum_contracts = []
    for dataset_path in dataset_paths:
        ethereum_contracts.append(pd.read_csv(dataset_path, index_col=None, header=0))
    ethereum_contracts = pd.concat(ethereum_contracts, axis=0, ignore_index=True)
    ethereum_contracts.columns = ['address', 'bytecode']

    logger.debug(ethereum_contracts.head())

    # Get all verified contracts
    verified_contracts = pd.DataFrame(columns=['address', 'bytecode', 'tokens'])

    total = 0
    for i, row in ethereum_contracts.iterrows():
        if total > 1000:
            break
        if mining.sourcecode_exists(row['address']):
            logger.debug(f"{i}: Adding {row['address']}")
            verified_contracts['address'] = row['address']
            verified_contracts['bytecode'] = row['bytecode']
            total += 1

    logger.debug(verified_contracts.head())

    # Get tokens from ABI for each verified contract
    for i, address in verified_contracts.address.iterrows():
        verified_contracts['tokens'].append(
            dict(Counter(cleaning.pipeline.clean(
                 mining.get_all_names_from_address(address))
            )))

    # Save
    verified_contracts.to_csv(os.path.join(cfg.dataset_root, 'verified_contracts.csv'),
                              header=True,
                              index=False)


if __name__ == '__main__':
    collect_verified_contracts()
