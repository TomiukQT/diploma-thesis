import re


def clean_twitter_text(text):
    return remove_pattern_from_text(text, "@[\w]*")


def remove_pattern_from_text( text, pattern):
    r = re.findall(pattern, text)
    for i in r:
        text = re.sub(i, "", text)
    return text
