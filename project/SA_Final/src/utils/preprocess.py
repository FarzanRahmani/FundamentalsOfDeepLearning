# first we tried dadmatech tools but does not work well, so we implement our own preprocess

import re
import string

import nltk
from cleantext.clean import remove_emoji as clean_text_remove_emoji
from hazm import Normalizer as HazmNormalizer
from hazm import stopwords_list
from nltk.corpus import stopwords

__all__ = ["clean_text"]


# __all__ = ["combined_preprocess"]
# from dadmatools.normalizer import Normalizer

# we can use hazm for tokenization, stemming, lemmatization, and POS tagging
# we wanted to implement but did not have time (future works)
def replace_emojis(text):
    # Happy
    grin = 'خنده'
    laugh = 'خنده'
    happy = 'خوشحال'
    _text = re.sub(":D", grin, text)
    _text = re.sub(" (x|X)D", laugh, _text)
    _text = re.sub(":\)+", happy, _text)

    # Sad
    sad = 'ناراحت'
    annoyed = 'رنجیده'
    _text = re.sub(":\(+", sad, _text)
    _text = re.sub("-_+-", annoyed, _text)
    return _text


def remove_emojis(text):
    _text = clean_text_remove_emoji(text)
    return _text


def remove_url(text):
    _text = re.sub(r"https?:\S+", '', text)
    return _text


def remove_punc(text):
    _text = text.translate(str.maketrans('', '', string.punctuation)) # remove punctuations from text using string.punctuation
    persian_virgol = '،'  # noqa
    _text = _text.replace(persian_virgol, ' ')
    return _text


def remove_numbers(text):
    _text = re.sub(r'\d+', '', text)
    return _text


def remove_hashtags(text):
    _text = re.sub(r'#\S+', '', text)
    return _text


def remove_mentions(text):
    _text = re.sub(r'@\S+', '', text)
    return _text


def remove_duplicate_spaces(text):
    _text = " ".join(text.split())
    return _text


def clean_text(text) -> str:
    _text = remove_punc(
        remove_numbers(
            remove_mentions(
                remove_hashtags(
                    remove_duplicate_spaces(
                        remove_url(
                            remove_emojis(text)
                        )
                    )
                )
            )
        )
    )

    normalizer = HazmNormalizer() # we use hazm for normalizing the text (removing extra spaces, etc.) (e.g. "می‌روم" -> "می روم", "خوبییییییییییییییی؟" -> "خوبی", "خوبی" -> "خوبی؟")
    _text = normalizer.normalize(_text)

    return _text


# def combined_preprocess(text: str) -> str:
#     normalizer = Normalizer(full_cleaning=True)
#     normalizer.remove_stop_word = False  # if it's True, it reduces the accuracy
#     normalizer.remove_puncs = False  # we remove punctuations in clean_text function
#     normalized_text = normalizer.normalize(text)
#     return clean_text(normalized_text)


if __name__ == "__main__":
    uncleaned_persian_text = "امروز با بچه‌ها میخوایم بریم بیرون و بععععدش بریم سینما :D https://t.co/1234567890"
    cleaned_text = clean_text(uncleaned_persian_text)
    print(cleaned_text)

    print(stopwords_list())

    nltk.download('stopwords')
    print(stopwords.words('english'))

    print(remove_url('https://github.com/roshan-research/hazm hi there'))
    print(remove_url('https hi there'))

    # download model from: https://github.com/roshan-research/hazm
    # tagger = POSTagger(model='resources/pos_tagger.model')
    # tagger.tag(word_tokenize('ما بسیار کتاب می‌خوانیم'))
