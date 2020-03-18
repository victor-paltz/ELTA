import string

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from collections import Counter


spacy_nlp = spacy.load('fr_core_news_sm')
nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'[a-z]\w+')


def raw_to_tokens(raw_string):
    """
    Transform a raw string into a stentence with cleaned words
    """

    string = raw_string.lower()
    string = remove_punctuation(string)
    tokens = tokenizer.tokenize(string)
    tokens = remove_stopwords(tokens)
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


def remove_unfrequent_words(X_train, min_occurrence=10):

    word_counter = Counter()
    for text in X_train["designation"]:
        word_counter.update(text.split())

    unfrequent_words = []
    for word in word_counter:
        if word_counter[word] < min_occurrence:
            unfrequent_words.append(word)

    def update_sentence(text):
        return " ".join([word for word in text.split()
                         if word not in unfrequent_words])

    X_train['designation'] = [update_sentence(text)
                              for text in X_train['designation']]

    return X_train
