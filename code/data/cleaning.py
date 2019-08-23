import re


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


