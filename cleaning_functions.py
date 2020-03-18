import string

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer

spacy_nlp = spacy.load('fr_core_news_sm')
nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'[a-z]\w+')


def raw_to_tokens(raw_string):
    """
    Transform a raw string into a list of cleaned words
    """

    string = raw_string.lower()
    string = remove_punctuation(string)
    tokens = tokenizer.tokenize(string)
    tokens = remove_stopwords(tokens)
    #tokens = word_lemmeatizer(tokens)
    cleaned_text = word_stemmer(tokens)
    return cleaned_text


def remove_punctuation(text):

    no_punct = ''.join(
        [c if c not in string.punctuation else " " for c in text])

    for i in range(10):
        no_punct = no_punct.replace(str(i), ' ')

    return no_punct


def remove_stopwords(words):
    no_stop_word = [c for c in words if c not in stopwords.words('french')]
    return no_stop_word


def word_lemmeatizer(words):
    lem_words = [lemmatizer.lemmatize(word) for word in words]
    return lem_words


def word_stemmer(words):
    stem_text = ' '.join([stemmer.stem(word) for word in words])
    return stem_text
