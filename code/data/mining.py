from etherscan.contracts import Contract
from google.cloud import bigquery
import os
import json


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
        "../dl-for-blockchain-1f8fa2978dfe.json"
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
    for row in query_job:
        print(row[0], row[1])


def get_sourcecode_from_address(address):
    """
    Return sourcecode for address

    See: https://git.io/fjbQW
         https://cloud.google.com/docs/authentication/getting-started
         https://sebs.github.io/etherscan-api/
    """
    api_key = os.environ['ETHERSCAN_API_KEY']
    api = Contract(address=address,
                   api_key=api_key)
    sourcecode = api.get_sourcecode()
    return sourcecode


def get_abi_from_address(address):
    """
    Return ABI for address
    """
    api_key = os.environ['ETHERSCAN_API_KEY']
    api = Contract(address=address,
                   api_key=api_key)
    return api.get_abi()


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
            names.append(x['name'])
        except KeyError:
            pass
        try:
            inputs = x['inputs']
        except KeyError:
            pass
        else:
            for ip in inputs:
                try:
                    names.append(ip['name'])
                except KeyError:
                    pass
        try:
            outputs = x['outputs']
        except KeyError:
            pass
        else:
            for op in outputs:
                try:
                    names.append(op['name'])
                except KeyError:
                    pass
    return result


if __name__ == '__main__':
    # Sanity Checks
    names = get_all_names_from_address('0x06012c8cf97bead5deae237070f9587f8e7a266d')
    print(names, len(names))
