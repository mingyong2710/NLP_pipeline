import re
import pandas as pd
import gensim
import spacy
from gensim.utils import simple_preprocess
from gensim.models import Phrases, CoherenceModel
from gensim.models.phrases import Phraser
from gensim import corpora
from nltk.corpus import stopwords
import pyLDAvis
import pyLDAvis.gensim
from pprint import pprint


class LDATopicModeler:
    def __init__(self, csv_file, text_column='Comment', custom_stopwords=None):
        self.df = pd.read_csv(csv_file)
        self.text_column = text_column
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        # Initialize stopwords
        self.stop_words = set(stopwords.words('english'))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
        
        # Placeholders for models
        self.bigram_mod = None
        self.trigram_mod = None
        self.lda_model = None
        self.corpus = None
        self.id2word = None

    def _preprocess_text(self, text):
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = re.sub(r"\'", "", text)     # Remove apostrophes
        return text.strip()

    def _sent_to_words(self, sentences):
        for sentence in sentences:
            yield simple_preprocess(str(sentence), deacc=True)

    def _remove_stopwords(self, texts):
        return [[word for word in doc if word not in self.stop_words] 
               for doc in texts]

    def _make_ngrams(self, texts):
        # Build the bigram and trigram models
        bigram = Phrases(texts, min_count=5, threshold=100)
        trigram = Phrases(bigram[texts], threshold=100)
        
        # Create more efficient Phraser objects
        self.bigram_mod = Phraser(bigram)
        self.trigram_mod = Phraser(trigram)
        
        # Apply the models
        texts = [self.bigram_mod[doc] for doc in texts]
        texts = [self.trigram_mod[self.bigram_mod[doc]] for doc in texts]
        return texts

    def _lemmatize(self, texts, allowed_postags=['NOUN', 'VERB','ADJ', 'ADV']):
        texts_out = []
        for sent in texts:
            doc = self.nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc 
                            if token.pos_ in allowed_postags])
        return texts_out

    def preprocess_data(self):
        # Get and clean text data
        data = self.df[self.text_column].astype(str).tolist()
        data = [self._preprocess_text(sent) for sent in data]
        # Tokenize
        data_words = list(self._sent_to_words(data))
        
        # Remove stopwords
        data_words = self._remove_stopwords(data_words)
        
        # Form n-grams
        data_words = self._make_ngrams(data_words)
        
        # Lemmatize and filter by POS
        data_words = self._lemmatize(data_words)
        
        return data_words

    def train_lda(self, num_topics=5, passes=10, random_state=42):
        """Train LDA model on preprocessed data"""
        # Preprocess data
        processed_texts = self.preprocess_data()
        
        # Create dictionary and corpus
        self.id2word = corpora.Dictionary(processed_texts)
        self.corpus = [self.id2word.doc2bow(text) for text in processed_texts]
        
        # Train LDA model
        self.lda_model = gensim.models.ldamodel.LdaModel(corpus=self.corpus,
                                           id2word=self.id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        
        return self.lda_model

    def visualize_topics(self):
        if not self.lda_model:
            raise ValueError("Model not trained yet. Call train_lda() first.")
            
        return pyLDAvis.gensim.prepare(
            self.lda_model, 
            self.corpus, 
            self.id2word,
            R=10
        )

if __name__ == "__main__":
    # Define custom stopwords
    custom_stopwords = [
        'thanks', 'thank', 'please', 'hi', 'hello', 'ok', 'bye',
        'uh', 'oh', 'hmm', 'lol', 'haha', 'omg', 'wow',
        'btw', 'idk', 'imo', 'tbh', 'u', 'ur', 'r', 'ya',
        'click', 'subscribe', 'follow', 'buy', 'check',
        'subject', 're', 'edu', 'com', 'www', 'http', 'html'
    ]
    
    # Initialize and process
    modeler = LDATopicModeler(
        csv_file='ebay_reviews.csv',
        text_column='Comment',
        custom_stopwords=custom_stopwords
    )
    
    # Train model
    lda_model = modeler.train_lda(num_topics=5)
    topics = lda_model.show_topics(num_topics=5, num_words=10, formatted=True)
    topics_df = pd.DataFrame(topics, columns=['Topic ID', 'Top Words'])
    topics_df.to_csv('lda_topics.csv', index=False)
    print(topics_df)

    # Visualize
    vis = modeler.visualize_topics()
    pyLDAvis.save_html(vis, 'lda_visualization.html')
