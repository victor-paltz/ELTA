from cleaning_functions import raw_to_tokens

test_sentences = [("il fait beau, l'etang est bleu", "fait beau etang bleu")]


def test_cleaning():
    for not_clean, clean in test_sentences:
        print(raw_to_tokens(not_clean), "|", clean)
        assert clean == raw_to_tokens(not_clean)
