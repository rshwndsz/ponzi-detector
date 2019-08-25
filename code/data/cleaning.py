import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    import nltk
    nltk.download('stopwords')

try:
    tokens = word_tokenize('Hello world')
except LookupError:
    import nltk
    nltk.download('punkt')


class CleaningPipeline:
    def __init__(self, func_list):
        self.func_list = func_list
        self.result = []

    def clean(self, string):
        self.result = self.func_list[0](string)
        for func in self.func_list[1:]:
            logger.info(f'CleaningPipeline: {func.__name__}({self.result})')
            self.result = func(self.result)
        return self.result


def comment_remover(string):
    """
    Remove comments from C style sourcecode
    :param string: C style sourcecode
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
    return re.sub(pattern, replacer, string)


def camel_case_split(string_list):
    """
    Split camel case into list

    :param string_list: list of words
    :return: list of split words in lowercase

    See: https://stackoverflow.com/a/29920015
    """
    result = []
    pattern = r'.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)'
    for string in string_list:
        matches = re.finditer(pattern, string)
        result.extend([m.group(0).lower() for m in matches if m.group(0)])
    return result


def snake_case_split(string_list):
    """
    Split snake case into list

    :param string_list: list of dirty strings
    :return: list of cleaned strings in lowercase
    """
    result = []
    for string in string_list:
        result.extend([x.lower() for x in string.split('_') if x])
    return result


def remove_common_words(string_list):
    return [w for w in string_list if w not in stop_words]


def remove_numbers_and_symbols(string_list):
    result = [re.sub('[^a-z]+', '', x) for x in string_list]
    return [x for x in result if x]


def stemming(string_list):
    ps = PorterStemmer()
    return [ps.stem(i) for i in string_list]


pipeline = CleaningPipeline([camel_case_split,
                             snake_case_split,
                             remove_common_words,
                             remove_numbers_and_symbols,
                             stemming,
                             ])


if __name__ == '__main__':
    # Sanity check
    from code.data.mining import get_all_names_from_address
    names = get_all_names_from_address('0x06012c8cf97bead5deae237070f9587f8e7a266d')
    print(pipeline.clean(names))
