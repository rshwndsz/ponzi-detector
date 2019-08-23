from etherscan.contracts import Contract
import os


def get_sourcecode(address):
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
