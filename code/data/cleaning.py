import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))


def comment_remover(text):
    """
    Remove comments from C style sourcecode
    :param text: C style sourcecode
    :return: sourcecode without comments
    """
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "      # note: a space and not an empty string
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)


def camel_case_split(text):
    """
    Split camel case into list

    :return: list of words

    See: https://stackoverflow.com/a/29920015
    """
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', text)
    return [m.group(0) for m in matches]


def snake_case_split(text):
    """
    Split snake case into list

    :return: list
    """
    return text.split('_')


def remove_common_words(string):
    if type(string) == list:
        return filter(lambda w: w not in stop_words, string)
    elif type(string) == str:
        return filter(lambda w: w not in stop_words, string.split(' '))
    else:
        raise TypeError('Must be string or list')


def remove_numbers_and_symbols(string):
    if type(string) == list:
        return [i for i in string if i.isalpha()]


def stemming(string_list):
    ps = PorterStemmer()
    return [ps.stem(i) for i in string_list]

