import regex as re
import spacy
spacy_nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])

class PreProcessing():

    def __init__(self, lowercase=True, remove_stopwords=True):
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        if self.remove_stopwords:
            stopwords = spacy.lang.en.stop_words.STOP_WORDS
            #print('Number of stop words: %d' % len(stopwords)) # 305

    def load_corpus(self, files):
        corpus = []
        for file in files:
            doc = self.load_file(file)
            corpus.extend(doc)
        return corpus

    def load_file(self, PATH, max_len=10000):
        doc = []
        with open(PATH, mode='r', encoding='utf8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line == "": continue
                sentence = self._normalize(line)
                doc.append(sentence)
                if i > max_len: break
        return doc

    def _normalize(self, text):
        # 1. lowercase
        if self.lowercase:
            text = text.lower()

        # 2. remove stopwords
        text = spacy_nlp(text)
        if self.remove_stopwords:
            tokens = [token.text for token in text if not token.is_stop]
        else:
            tokens = [token.text for token in text]

        # 3. remove file infos
        tokens = self._remove_file_infos(tokens)

        # 4. add starting "\u0120" token
        text = " ".join(tokens)

        return text

    def _remove_file_infos(self, tokens):
        tokens = [token for token in tokens if not "\x00" in token]
        return tokens


def corpus():
    p = PreProcessing(lowercase=False, remove_stopwords=False)
    PATH = "./data/urlsf_subset00-1_data"
    doc = p.load_file(PATH, max_len=10000)
    return doc
