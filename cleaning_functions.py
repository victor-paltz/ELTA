import string

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer

spacy_nlp = spacy.load('fr_core_news_sm')
nltk.download('wordnet')
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
    tokens = word_lemmeatizer(tokens)
    cleaned_text = word_stemmer(tokens)
    return cleaned_text


def remove_punctuation(text):
    no_punct = ''.join([c for c in text if c not in string.punctuation])
    no_punct = no_punct.replace('0', ' ')
    no_punct = no_punct.replace('1', ' ')
    no_punct = no_punct.replace('2', ' ')
    no_punct = no_punct.replace('3', ' ')
    no_punct = no_punct.replace('4', ' ')
    no_punct = no_punct.replace('5', ' ')
    no_punct = no_punct.replace('6', ' ')
    no_punct = no_punct.replace('7', ' ')
    no_punct = no_punct.replace('8', ' ')
    no_punct = no_punct.replace('9', ' ')
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
