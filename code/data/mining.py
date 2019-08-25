from etherscan.contracts import Contract
from google.cloud import bigquery
import os
import json
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def test_client_creation():
    try:
        from google.cloud import bigquery
    except ImportError:
        raise ImportError('Please install google-cloud-python')

    else:
        client = bigquery.Client
        assert client is not None


def get_all_addresses(limit=1000):
    """
    Get all addresses from Google BigQuery

    Query bigquery-public-data:crypto_ethereum.transactions
    for bytecode and addresses
    See: https://googleapis.github.io/google-cloud-python/latest/
    """
    client = bigquery.Client.from_service_account_json(
        "../secrets/dl-for-blockchain-1f8fa2978dfe.json"
    )
    assert client is not None
    query = (
        "SELECT address, bytecode "
        "FROM `bigquery-public-data.crypto_ethereum.contracts` "
        "WHERE bytecode is not null "
        f"LIMIT {limit}"
    )
    query_job = client.query(
        query,
        location="US"
    )
    return query_job


def get_bytecode_from_addresses(addresses):
    """
        Get all addresses from Google BigQuery
        Query bigquery-public-data:crypto_ethereum.transactions
        for bytecode and addresses
        See: https://googleapis.github.io/google-cloud-python/latest/
        """
    client = bigquery.Client.from_service_account_json(
        "../secrets/My Project 14448-c3df0e5d3ec5.json"
    )
    assert client is not None
    query = (
        "SELECT bytecode "
        "FROM `bigquery-public-data.crypto_ethereum.contracts` "
        f"WHERE address in {tuple(addresses)}"
    )
    query_job = client.query(
        query,
        location="US"
    )
    return query_job


def get_etherscan_api_key(path='../secrets/etherscan_api_key.txt'):
    with open(path, 'r') as f:
        api_key = f.readline().strip()
    return api_key


def get_sourcecode_from_address(address):
    """
    Return sourcecode for address

    See: https://git.io/fjbQW
         https://cloud.google.com/docs/authentication/getting-started
         https://sebs.github.io/etherscan-api/
    """
    api_key = get_etherscan_api_key()
    api = Contract(address=address,
                   api_key=api_key)
    try:
        sourcecode = api.get_sourcecode()
    except:
        logger.warning('Empty Response. Try again later.')
        return None
    else:
        if not sourcecode[0]['SourceCode']:
            return None
        if sourcecode[0]['ABI'] == 'Contract source code not verified':
            return None
        return sourcecode


def get_abi_from_address(address):
    """
    Return ABI for address
    """
    api_key = get_etherscan_api_key()
    api = Contract(address=address,
                   api_key=api_key)
    try:
        abstract_bi = api.get_abi()
    except:
        logger.warning('Empty Response. Try again later.')
        return None
    else:
        if not abstract_bi:
            return None
        return abstract_bi


def sourcecode_exists(address):
    request_result = get_sourcecode_from_address(address)
    if request_result:
        return True
    return False


def get_all_names_from_address(address):
    """
    Gets all names from a contract's ABI

    The names are not cleaned here
    :param address: Address of contract
    :return: List of identifiers/names
    """
    abi = json.loads(get_abi_from_address(address))
    result = []
    for x in abi:
        try:
            result.append(x['name'])
        except KeyError:
            pass
        try:
            inputs = x['inputs']
        except KeyError:
            pass
        else:
            for ip in inputs:
                try:
                    result.append(ip['name'])
                except KeyError:
                    pass
        try:
            outputs = x['outputs']
        except KeyError:
            pass
        else:
            for op in outputs:
                try:
                    result.append(op['name'])
                except KeyError:
                    pass
    return result


if __name__ == '__main__':
    # Sanity Checks
    names = get_all_names_from_address('0x06012c8cf97bead5deae237070f9587f8e7a266d')
    print(names, len(names))

    from collections import Counter
    from code.data.cleaning import pipeline
    cleaned_names = pipeline.clean(names)
    hash_map = dict(Counter(cleaned_names))
    print(hash_map)
