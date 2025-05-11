# LDA.py

import re
import pandas as pd
import gensim
import spacy
from gensim.utils import simple_preprocess
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim import corpora
import nltk
from nltk.corpus import stopwords
import logging # For more structured logging if you prefer over print

# Configure basic logging (optional, print statements are also used for simplicity here)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure stopwords are available
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    print("NLTK stopwords downloaded.")

class LDATopicModeler:
    def __init__(self, df_docs, text_column='Comment', custom_stopwords_list=None):
        """
        Initializes the modeler with a DataFrame.
        Args:
            df_docs (pd.DataFrame): The DataFrame containing the text data.
            text_column (str): The name of the column in df_docs that contains the text.
            custom_stopwords_list (list, optional): A list of custom stopwords.
        """
        print(f"[LDATopicModeler __init__] Initializing with {len(df_docs)} documents from column '{text_column}'.")
        if not isinstance(df_docs, pd.DataFrame):
            raise ValueError("Input df_docs must be a pandas DataFrame.")
        if text_column not in df_docs.columns:
            raise ValueError(f"Text column '{text_column}' not found in DataFrame.")

        self.df_docs = df_docs.copy() # Work on a copy
        self.text_column = text_column
        
        try:
            self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
            print("[LDATopicModeler __init__] SpaCy model 'en_core_web_sm' loaded.")
        except OSError as e:
            print(f"[LDATopicModeler __init__] ERROR: Failed to load SpaCy model: {e}")
            raise # Reraise the error to be caught by Streamlit if applicable
            
        self.stop_words = set(stopwords.words('english'))
        if custom_stopwords_list:
            self.stop_words.update(custom_stopwords_list)
            print(f"[LDATopicModeler __init__] Updated stopwords. Total: {len(self.stop_words)}.")
        
        # Model components
        self.bigram_mod = None
        self.trigram_mod = None
        self.lda_model = None
        self.corpus = None
        self.id2word = None
        
        # Processed data
        self.original_texts = [] # Store original texts corresponding to processed_texts
        self.processed_texts_tokens = [] # List of token lists
        self.doc_indices_after_preprocessing = [] # Indices from original df_docs that survived initial preprocessing

    def _clean_text_basic(self, text):
        text = str(text) # Ensure text is string
        text = re.sub(r'\s+', ' ', text) # Remove extra whitespace
        text = re.sub(r"\'", "", text)     # Remove apostrophes (simple approach)
        text = text.lower() # Convert to lowercase
        return text.strip()

    def _tokenize_sentences(self, sentences):
        for sentence in sentences:
            # deacc=True removes accents
            yield simple_preprocess(str(sentence), deacc=True, min_len=2, max_len=15)

    def _filter_stopwords(self, list_of_token_lists):
        return [[word for word in doc_tokens if word not in self.stop_words] 
               for doc_tokens in list_of_token_lists]

    def _apply_ngrams(self, list_of_token_lists, min_ngram_count=5, ngram_threshold=100):
        if not list_of_token_lists or not any(list_of_token_lists):
            print("[LDATopicModeler _apply_ngrams] Input token lists are empty. Skipping n-grams.")
            return list_of_token_lists # Return as is if input is empty

        # Filter out empty lists before feeding to Phrases, as it can cause issues
        non_empty_token_lists = [doc for doc in list_of_token_lists if doc]
        if not non_empty_token_lists:
            print("[LDATopicModeler _apply_ngrams] All token lists became empty before n-gram. Skipping n-grams.")
            return list_of_token_lists


        bigram_phraser = Phrases(non_empty_token_lists, min_count=min_ngram_count, threshold=ngram_threshold)
        trigram_phraser = Phrases(bigram_phraser[non_empty_token_lists], min_count=min_ngram_count, threshold=ngram_threshold)
        
        # Use Phraser for efficiency if these models are to be reused (they are not here, but good practice)
        # self.bigram_mod = Phraser(bigram_phraser)
        # self.trigram_mod = Phraser(trigram_phraser)

        # Apply to the original list_of_token_lists to maintain original document count if some were empty
        # This is tricky because Phrases only works on non-empty. We need to map back.
        # For simplicity, let's process non_empty and then rebuild, or process one by one.
        # The Phraser object is better for applying to individual documents if structure is complex.

        # Simpler application for this flow:
        texts_with_bigrams = [bigram_phraser[doc] for doc in list_of_token_lists if doc] # Apply only to non-empty
        texts_with_trigrams = [trigram_phraser[doc] for doc in texts_with_bigrams if doc] # Apply only to non-empty
        
        # Reconstruct the full list, keeping empty lists where they were
        result_list = []
        processed_idx = 0
        for original_doc_tokens in list_of_token_lists:
            if original_doc_tokens and processed_idx < len(texts_with_trigrams) :
                result_list.append(texts_with_trigrams[processed_idx])
                processed_idx += 1
            else: # If original was empty or became empty and was skipped by Phrases
                result_list.append([])
        print(f"[LDATopicModeler _apply_ngrams] N-grams applied. Input docs: {len(list_of_token_lists)}, Output docs: {len(result_list)}")
        return result_list


    def _lemmatize_and_pos_filter(self, list_of_token_lists, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        lemmatized_texts = []
        if not list_of_token_lists:
            return []
        for doc_tokens in list_of_token_lists:
            if not doc_tokens: # Handle empty list of tokens for a document
                lemmatized_texts.append([])
                continue
            # Join tokens to form a string for SpaCy's nlp object
            doc_text_for_spacy = " ".join(doc_tokens)
            spacy_doc = self.nlp(doc_text_for_spacy)
            lemmatized_texts.append([token.lemma_ for token in spacy_doc 
                                     if token.pos_ in allowed_postags and not token.is_stop and token.is_alpha])
        print(f"[LDATopicModeler _lemmatize_and_pos_filter] Lemmatization complete. Input docs: {len(list_of_token_lists)}, Output docs: {len(lemmatized_texts)}")
        return lemmatized_texts

    def preprocess_documents(self):

        print("[LDATopicModeler preprocess_documents] Starting preprocessing...")
        self.original_texts = self.df_docs[self.text_column].astype(str).tolist()
        
        # 1. Basic Cleaning
        cleaned_texts = [self._clean_text_basic(text) for text in self.original_texts]
        print(f"[LDATopicModeler preprocess_documents] Step 1: Basic cleaning done. Count: {len(cleaned_texts)}")

        # 2. Tokenization
        tokenized_docs = list(self._tokenize_sentences(cleaned_texts))
        print(f"[LDATopicModeler preprocess_documents] Step 2: Tokenization done. Count: {len(tokenized_docs)}")

        # 3. Stopword Filtering
        docs_no_stopwords = self._filter_stopwords(tokenized_docs)
        print(f"[LDATopicModeler preprocess_documents] Step 3: Stopword removal done. Count: {len(docs_no_stopwords)}")

        # 4. N-gram Formation
        # Note: N-grams are often applied *before* lemmatization to catch phrases like "new_york"
        docs_with_ngrams = self._apply_ngrams(docs_no_stopwords)
        print(f"[LDATopicModeler preprocess_documents] Step 4: N-gram formation done. Count: {len(docs_with_ngrams)}")

        # 5. Lemmatization and POS Tag Filtering
        final_processed_tokens = self._lemmatize_and_pos_filter(docs_with_ngrams)
        print(f"[LDATopicModeler preprocess_documents] Step 5: Lemmatization done. Count: {len(final_processed_tokens)}")
        
        # Store results and filter out completely empty documents for LDA training
        self.processed_texts_tokens = []
        self.doc_indices_after_preprocessing = [] # Original indices of docs that are NOT empty

        for i, tokens in enumerate(final_processed_tokens):
            if tokens: # Only keep documents that still have tokens
                self.processed_texts_tokens.append(tokens)
                self.doc_indices_after_preprocessing.append(i) # Store original index
        
        print(f"[LDATopicModeler preprocess_documents] Preprocessing finished. {len(self.processed_texts_tokens)} non-empty documents remaining out of {len(self.original_texts)} original.")
        if not self.processed_texts_tokens:
            print("[LDATopicModeler preprocess_documents] WARNING: All documents became empty after preprocessing.")
        
        return self.processed_texts_tokens


    def train_lda_model(self, num_topics=10, passes=10, random_state=100, 
                        dict_no_below=5, dict_no_above=0.5, workers=None):

        print(f"\n[LDATopicModeler train_lda_model] Attempting to train LDA with {num_topics} topics.")
        
        if not self.processed_texts_tokens or not any(self.processed_texts_tokens):
            print("[LDATopicModeler train_lda_model] ERROR: No preprocessed texts available or all are empty. Cannot train model.")
            self.preprocess_documents() # Attempt to preprocess if not done
            if not self.processed_texts_tokens or not any(self.processed_texts_tokens):
                 print("[LDATopicModeler train_lda_model] ERROR: Still no preprocessed texts after attempting. Aborting training.")
                 return None

        # 1. Create Dictionary
        self.id2word = corpora.Dictionary(self.processed_texts_tokens)
        print(f"[LDATopicModeler train_lda_model] Initial dictionary size: {len(self.id2word)} words.")
        
        # 2. Filter Dictionary Extremes
        self.id2word.filter_extremes(no_below=dict_no_below, no_above=dict_no_above)
        print(f"[LDATopicModeler train_lda_model] Dictionary size after filtering (no_below={dict_no_below}, no_above={dict_no_above}): {len(self.id2word)} words.")
        
        if not self.id2word:
            print("[LDATopicModeler train_lda_model] ERROR: Dictionary became empty after filtering. Cannot train model. Try adjusting filter_extremes values.")
            return None

        # 3. Create Corpus (Bag-of-Words for each document)
        # Ensure corpus is built only from tokens that exist in the filtered dictionary
        temp_corpus = [self.id2word.doc2bow(text_tokens) for text_tokens in self.processed_texts_tokens]
        
        # Filter out documents that became empty after dictionary mapping
        self.corpus = []
        final_doc_indices_for_lda = [] # Original indices of docs that are in the final corpus
        
        for i, bow in enumerate(temp_corpus):
            if bow: # If Bag-of-Words is not empty
                self.corpus.append(bow)
                # `i` here is the index within `self.processed_texts_tokens`
                # We need to map it back to the original df_docs index via `self.doc_indices_after_preprocessing`
                final_doc_indices_for_lda.append(self.doc_indices_after_preprocessing[i])

        self.doc_indices_in_corpus = final_doc_indices_for_lda # Store for get_dominant_topics_df

        print(f"[LDATopicModeler train_lda_model] Corpus created. Number of documents in corpus: {len(self.corpus)}.")
        
        if not self.corpus:
            print("[LDATopicModeler train_lda_model] ERROR: Corpus is empty. No documents contained words from the filtered dictionary. Cannot train model.")
            return None
        
        print(f"[LDATopicModeler train_lda_model] Training LDA model (this may take some time)...")
        
        lda_params = {
            'corpus': self.corpus,
            'id2word': self.id2word,
            'num_topics': num_topics,
            'random_state': random_state,
            'update_every': 1, # Update model after processing each chunk
            'chunksize': 100,  # Number of documents to process at once
            'passes': passes,
            'alpha': 'auto',   # Learn alpha from data
            'eta': 'auto',     # Learn eta from data
            'per_word_topics': True # Essential for pyLDAvis and some topic analyses
        }
        if workers is not None:
            lda_params['workers'] = workers # For LdaMulticore

        try:
            # Use LdaMulticore if workers are specified and > 1, otherwise LdaModel
            if workers and workers > 1:
                 print(f"[LDATopicModeler train_lda_model] Using LdaMulticore with {workers} workers.")
                 self.lda_model = gensim.models.ldamulticore.LdaMulticore(**lda_params)
            else:
                 print("[LDATopicModeler train_lda_model] Using LdaModel (single core).")
                 if 'workers' in lda_params: del lda_params['workers'] # LdaModel doesn't take 'workers'
                 self.lda_model = gensim.models.ldamodel.LdaModel(**lda_params)
            
            print("[LDATopicModeler train_lda_model] LDA model training completed successfully.")
            return self.lda_model
        except Exception as e:
            print(f"[LDATopicModeler train_lda_model] ERROR: Exception during LDA model training: {e}")
            import traceback
            traceback.print_exc()
            self.lda_model = None
            return None
    def get_dominant_topics_dataframe(self, min_topic_probability=0.01):
        """
        Assigns a dominant topic to each document that was part of the LDA corpus
        and provides information about other documents.
        Returns:
            pd.DataFrame: DataFrame with Document_ID (original index), Original_Text_Snippet,
                          Dominant_Topic, Probability, and Top_Words_Dominant_Topic.
        """
        print("[LDATopicModeler get_dominant_topics_dataframe] Generating dominant topics...")
        if not self.lda_model:
            print("[LDATopicModeler get_dominant_topics_dataframe] ERROR: LDA model not trained. Cannot get dominant topics.")
            return None
        if self.df_docs is None:
            print("[LDATopicModeler get_dominant_topics_dataframe] ERROR: Original DataFrame (self.df_docs) is not available.")
            return None

        all_doc_topic_info = []

        # Process documents that are in the LDA corpus
        for i, corpus_doc_bow in enumerate(self.corpus):
            original_doc_idx = self.doc_indices_in_corpus[i] # Get original index
            doc_topics = self.lda_model.get_document_topics(corpus_doc_bow, minimum_probability=min_topic_probability)
            
            text_snippet = str(self.df_docs[self.text_column].iloc[original_doc_idx])[:150] + "..."

            if doc_topics:
                dominant_topic_id, prob = sorted(doc_topics, key=lambda x: x[1], reverse=True)[0]
                topic_words_list = self.lda_model.show_topic(dominant_topic_id, topn=5) # Get top 5 words
                topic_keywords = ", ".join([word for word, _ in topic_words_list])
                
                all_doc_topic_info.append({
                    'Document_ID': original_doc_idx,
                    'Original_Text_Snippet': text_snippet,
                    'Dominant_Topic': dominant_topic_id,
                    'Probability': round(prob, 4),
                    'Top_Words_Dominant_Topic': topic_keywords,
                    'Status': 'In LDA Corpus'
                })
            else: # Document in corpus but no topic met min_topic_probability
                all_doc_topic_info.append({
                    'Document_ID': original_doc_idx,
                    'Original_Text_Snippet': text_snippet,
                    'Dominant_Topic': -1, # No dominant topic found above threshold
                    'Probability': 0.0,
                    'Top_Words_Dominant_Topic': "N/A (Low Probability)",
                    'Status': 'In LDA Corpus - No Dominant Topic'
                })
        
        # Identify documents that were filtered out before LDA corpus creation
        original_indices_in_lda_corpus = set(self.doc_indices_in_corpus)
        for original_idx in range(len(self.df_docs)):
            if original_idx not in original_indices_in_lda_corpus:
                text_snippet = str(self.df_docs[self.text_column].iloc[original_idx])[:150] + "..."
                all_doc_topic_info.append({
                    'Document_ID': original_idx,
                    'Original_Text_Snippet': text_snippet,
                    'Dominant_Topic': -2, # Indicates filtered out before LDA
                    'Probability': 0.0,
                    'Top_Words_Dominant_Topic': "N/A (Filtered Out)",
                    'Status': 'Filtered Out Pre-LDA'
                })

        if not all_doc_topic_info:
            print("[LDATopicModeler get_dominant_topics_dataframe] WARNING: No topic information generated (empty result).")
            return pd.DataFrame() # Return empty DataFrame

        df_dominant_topics = pd.DataFrame(all_doc_topic_info)
        df_dominant_topics = df_dominant_topics.sort_values(by='Document_ID').reset_index(drop=True)
        print(f"[LDATopicModeler get_dominant_topics_dataframe] Dominant topics DataFrame created with {len(df_dominant_topics)} rows.")
        return df_dominant_topics
