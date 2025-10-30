import nltk
import spacy

nltk.download("vader_lexicon", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords")
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")
