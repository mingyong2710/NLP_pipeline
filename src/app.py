import streamlit as st
import pandas as pd
import time # For simulating work if needed

# Configure page before any other Streamlit calls
st.set_page_config(layout="wide", page_title="Multi-source Chatbot & Topic Explorer", page_icon="🤖")

# --- Helper/Utility Imports ---
try:
    from utils.openai_helper import get_openai_streaming_response, get_openai_response
except ImportError:
    st.error("Failed to import OpenAI helper. Ensure utils/openai_helper.py exists and is configured.")
    # Dummy functions if not found, so app can partially load
    def get_openai_streaming_response(messages): yield "OpenAI helper not found."
    def get_openai_response(prompt): return "OpenAI helper not found."
    
try:
    from ABSA import perform_absa_on_dataframe # Import the modified function
except ImportError:
    st.error("Failed to import ABSA.py or perform_absa_on_dataframe function.")
    def perform_absa_on_dataframe(df, col, proc_entire): return pd.DataFrame()

try:
    from crawl_ebay import get_ebay_reviews
    from crawl_youtube import get_youtube_comments
    from crawl_amazon import get_amazon_reviews
except ImportError:
    st.error("Failed to import one or more crawling scripts. Ensure they exist.")
    # Dummy functions
    def get_ebay_reviews(url, count, headless): return [{"comment": f"eBay crawl mock for {url}"}] * count
    def get_youtube_comments(url, count): return [f"YT crawl mock for {url}"] * count
    def get_amazon_reviews(url, count, headless): return [f"Amazon crawl mock for {url}"] * count


# --- Topic Modeling Imports ---
try:
    from LDA import LDATopicModeler
except ImportError:
    st.error("Failed to import LDATopicModeler. Ensure LDA.py exists and is correctly defined.")
    # Dummy class if not found
    class LDATopicModeler:
        def __init__(self, *args, **kwargs): pass
        def preprocess_documents(self): self.processed_texts_tokens = []
        def train_lda_model(self, *args, **kwargs): return None
        def get_dominant_topics_dataframe(self): return pd.DataFrame()

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import altair as alt
import numpy as np # Required by Altair for some type checks


# --- Streamlit Caching for Topic Modeling ---
@st.cache_resource
def get_topic_modeler_instance(df_for_modeler, text_col, custom_stopwords_list):
    """Gets or creates an LDATopicModeler instance."""
    # print("APP.PY CACHE: Creating/getting LDATopicModeler instance.") # For debugging
    # Ensure df_for_modeler is a DataFrame and text_col exists
    if not isinstance(df_for_modeler, pd.DataFrame) or text_col not in df_for_modeler.columns:
        st.error(f"Invalid DataFrame or text column ('{text_col}') for Topic Modeler.")
        # Return a dummy object or raise error, depending on desired handling
        return LDATopicModeler(pd.DataFrame({'text':['error']}), 'text', []) # Dummy to prevent further crashes

    modeler = LDATopicModeler(
        df_docs=df_for_modeler.copy(), # Use a copy
        text_column=text_col,
        custom_stopwords_list=custom_stopwords_list
    )
    return modeler

@st.cache_data()
def train_lda_and_get_details(_modeler_obj, num_topics, passes=10, random_state=100,
                              dict_no_below=1, dict_no_above=0.95):
    """Trains LDA and returns model components."""
    # print(f"APP.PY CACHE: train_lda_and_get_details for {_modeler_obj} with {num_topics} topics.") # For debugging
    
    # Ensure modeler object is valid
    if not hasattr(_modeler_obj, 'preprocess_documents') or not hasattr(_modeler_obj, 'train_lda_model'):
        st.error("Invalid modeler object passed to train_lda_and_get_details.")
        return None, None, None, None, None

    # Ensure preprocessing is done
    # The LDATopicModeler class should handle this internally or have this attribute
    if not getattr(_modeler_obj, 'processed_texts_tokens', None):
        # print("APP.PY CACHE: Preprocessed texts not found, calling preprocess_documents().")
        _modeler_obj.preprocess_documents()

    lda_model_trained = _modeler_obj.train_lda_model(
        num_topics=num_topics,
        passes=passes,
        random_state=random_state,
        dict_no_below=dict_no_below,
        dict_no_above=dict_no_above
    )

    if lda_model_trained:
        # print("APP.PY CACHE: LDA model trained. Getting dominant topics DF.") # For debugging
        dominant_topics_df = _modeler_obj.get_dominant_topics_dataframe()
        # df_docs should be an attribute of the _modeler_obj, set during its __init__
        original_df_used = getattr(_modeler_obj, 'df_docs', pd.DataFrame()) 
        return (
            lda_model_trained,
            getattr(_modeler_obj, 'corpus', None),
            getattr(_modeler_obj, 'id2word', None),
            dominant_topics_df,
            original_df_used # Return the DataFrame that the modeler was initialized with
        )
    # print("APP.PY CACHE: LDA model training failed or returned None.") # For debugging
    return None, None, None, None, None

@st.cache_data()
def get_absa_results(_df_input, _text_column, _process_entire_review=False):
    """ Caches the ABSA results for a given DataFrame and text column. """
    if _df_input is None or _df_input.empty:
        return pd.DataFrame()
    print(f"APP.PY CACHE: Calling perform_absa_on_dataframe for DataFrame with shape {_df_input.shape} on column '{_text_column}'")
    return perform_absa_on_dataframe(_df_input, _text_column, _process_entire_review)



# --- Main App Logic ---
def main():
    #st.sidebar.image("https://uploads-ssl.webflow.com/62320769738609302808ab93/6233500472781c1811059112_Logo-LKN-noi-dung-trang-p-500.png", width=100)
    st.sidebar.title("NLP Pipeline")
    st.sidebar.write("Empowering Insights from Text Data")
    st.sidebar.markdown("---")


    # Initialize topic_model_data in session state if it's not there
    if 'topic_model_data' not in st.session_state:
        st.session_state.topic_model_data = {
            "df": None, "text_col": None, "lda_model": None, "corpus": None,
            "id2word": None, "dominant_topics_df": None, 
            "df_display_for_topic_model": None, # The df the model was actually trained on
            "source_info": None 
        }
    if 'absa_results_data' not in st.session_state: # NEW for ABSA
        st.session_state.absa_results_data = {
            "results_df": None,
            "source_info": None # To know which data the ABSA was run on
        }
        
    # Add "ABSA Explorer" to mode selection
    modes = ["Chatbot", "Topic Modeling Explorer", "ABSA Results Explorer"]
    mode = st.sidebar.selectbox("Select Functionality", modes)

    if mode == "Chatbot":
        run_chatbot()
    elif mode == "Topic Modeling Explorer":
        run_topic_modeling_interactive()
    elif mode == "ABSA Results Explorer": # NEW
        run_absa_results_explorer()
    else:
        st.error("Invalid mode selected.")

def run_chatbot():
    st.sidebar.subheader("Chatbot Configuration")
    if 'conversation' not in st.session_state: st.session_state.conversation = []
    if 'messages_history' not in st.session_state: st.session_state.messages_history = [{'role': 'system', 'content': 'You are a helpful AI assistant.'}]
    if 'current_items' not in st.session_state: st.session_state.current_items = [] # For raw scraped items
    if 'current_scraped_texts' not in st.session_state: st.session_state.current_scraped_texts = [] # For LDA-ready texts


    source = st.sidebar.selectbox("Select Data Source", ["Amazon", "eBay", "YouTube"], key="chatbot_source_select")
    url = st.sidebar.text_input("Enter Product/Video URL", key="chatbot_url_input")
    num_comments = st.sidebar.slider("Number of Comments/Reviews", 1, 100, 20, key="chatbot_num_comments_slider") # Increased max

    run_lda_on_scraped = st.sidebar.checkbox("Also run Topic Modeling on scraped data?", True, key="chatbot_run_lda_checkbox")
    if run_lda_on_scraped:
        num_topics_for_scraped = st.sidebar.slider("Number of Topics (for scraped data)", 2, 15, 3, key="chatbot_lda_topics_slider")
        custom_stopwords_scraped_str = st.sidebar.text_area(
            "Custom stopwords (for scraped data LDA)",
            value="thanks,thank,please,hi,hello,ok,buy,subscribe,follow,click,subject,re,edu,com,www,http,html,product,item,review,video,channel,comment,watch,view,like,dislike,great,good,bad,awesome",
            key="chatbot_lda_stopwords_area"
        )
        
    run_absa_on_scraped = st.sidebar.checkbox("Run ABSA (Aspect Sentiment)?", True, key="chatbot_run_absa_cb_v2") # NEW
    if run_absa_on_scraped:
        absa_processing_mode = st.sidebar.radio(
            "ABSA Processing:", ("Sentence-Level", "Entire Review"), index=0, key="chatbot_absa_mode_radio"
        )
        process_entire_review_for_absa = (absa_processing_mode == "Entire Review")

    if st.sidebar.button("Scrape & Analyze", key="chatbot_scrape_analyze_button"):
        if not url:
            st.sidebar.error("Please enter a URL!")
        else:
            st.session_state.conversation = []
            st.session_state.current_scraped_texts = [] # Reset

            with st.spinner(f"Crawling data from {source}..."):
                items_raw = [] # Raw data from crawler
                processed_texts_for_lda = [] # Cleaned list of strings for LDA

                if source == "Amazon":
                    items_raw = get_amazon_reviews(url, num_comments, False)
                    # Assuming get_amazon_reviews returns list of strings or dicts with a known text key
                    # Adapt this based on actual return type of your crawler
                    if items_raw and isinstance(items_raw[0], str):
                        processed_texts_for_lda = items_raw
                    elif items_raw and isinstance(items_raw[0], dict):
                        processed_texts_for_lda = [d.get('review_text', d.get('text', str(d))) for d in items_raw] # Example keys
                elif source == "eBay":
                    items_raw = get_ebay_reviews(url, num_comments, False)
                    processed_texts_for_lda = [item['comment'] for item in items_raw if isinstance(item, dict) and 'comment' in item]
                elif source == "YouTube":
                    items_raw = get_youtube_comments(url, num_comments)
                    processed_texts_for_lda = items_raw # Assuming it returns list of strings

                st.session_state.current_items = items_raw # Store raw items
                st.session_state.current_scraped_texts = processed_texts_for_lda # Store texts for LDA

                if not processed_texts_for_lda:
                    st.sidebar.warning("No text content was successfully scraped or extracted.")
                    st.rerun() # Stop further processing if no text

                st.sidebar.success(f"Successfully scraped {len(processed_texts_for_lda)} items from {source}.")
                
            df_scraped = pd.DataFrame(processed_texts_for_lda, columns=['text_content'])
            text_col_scraped = 'text_content'

            # --- OpenAI Summary ---
            if processed_texts_for_lda:
                with st.spinner("Generating OpenAI summary..."):
                    summary_prompt_parts = [f"Here are user comments/reviews from {source} regarding {url}:\n"]
                    for i, text_content in enumerate(processed_texts_for_lda[:50], start=1): # Limit for prompt
                        summary_prompt_parts.append(f"{i}. {text_content}\n")
                    summary_prompt = "".join(summary_prompt_parts)
                    summary = get_openai_response(summary_prompt + "\nSummarize the key points and overall sentiment.")
                    
                    st.session_state.conversation.append(('assistant', f"Data scraped from {source} ({url})!\n**AI Summary:**\n{summary}"))
                    system_message_content = f"You are an AI assistant. Context: User comments/reviews from {source} for {url}.\nSummary: {summary}\nOriginal data sample:\n{''.join(summary_prompt_parts)}"
                    st.session_state.messages_history = [{'role': 'system', 'content': system_message_content}]

            # --- Automated LDA Topic Modeling on Scraped Data ---
            if run_lda_on_scraped and processed_texts_for_lda:
                with st.spinner("Performing Topic Modeling on scraped data..."):
                    df_scraped_for_lda = pd.DataFrame(processed_texts_for_lda, columns=['text_content'])
                    text_col_for_scraped_lda = 'text_content'
                    custom_stopwords_list_scraped = [w.strip() for w in custom_stopwords_scraped_str.split(',') if w.strip()]

                    modeler_scraped = get_topic_modeler_instance(
                        df_scraped_for_lda, text_col_for_scraped_lda, custom_stopwords_list_scraped
                    )
                    
                    l_lda, l_corpus, l_id2word, l_dom_df, l_df_display = train_lda_and_get_details(
                        modeler_scraped, num_topics_for_scraped,
                        dict_no_below=1, dict_no_above=0.95 # Sensible defaults for scraped
                    )

                    if l_lda:
                        st.session_state.topic_model_data.update({
                            "lda_model": l_lda, "corpus": l_corpus, "id2word": l_id2word,
                            "dominant_topics_df": l_dom_df, 
                            "df_display_for_topic_model": l_df_display, # This is df_scraped_for_lda
                            "text_col": text_col_for_scraped_lda,
                            "source_info": f"Scraped from {source}: {url.split('?')[0]}" # Shorter URL
                        })
                        st.sidebar.success("Topic Modeling on scraped data complete!")
                        brief_topic_summary = ["\n**Key Topics Found (from scraped data):**"]
                        for i in range(min(l_lda.num_topics, 3)): # Show top 3 topics
                            words = [word for word, prob in l_lda.show_topic(i, topn=3)]
                            brief_topic_summary.append(f"- Topic {i+1}: {', '.join(words)}")
                        st.session_state.conversation.append(('assistant', "\n".join(brief_topic_summary) +
                                                             "\n(Explore these topics in the 'Topic Modeling Explorer' tab.)"))
                    else:
                        st.sidebar.error("Failed to train topic model on scraped data.")
                        st.session_state.topic_model_data["lda_model"] = None # Clear if failed

            # Save scraped data (optional)
            if processed_texts_for_lda:
                df_to_save = pd.DataFrame(processed_texts_for_lda, columns=['text_content'])
                filename_save = f"{source.lower()}_scraped_data.csv"
                csv_data = df_to_save.to_csv(index=False).encode('utf-8')
                st.sidebar.download_button("Download Scraped Data (CSV)", csv_data, filename_save, 'text/csv', key="download_scraped_csv_button")

            if run_absa_on_scraped and not df_scraped.empty:
                with st.spinner("Performing Aspect-Based Sentiment Analysis..."):
                    absa_df_results = get_absa_results(df_scraped, text_col_scraped, process_entire_review_for_absa)
                    
                    if absa_df_results is not None and not absa_df_results.empty:
                        st.session_state.absa_results_data["results_df"] = absa_df_results
                        st.session_state.absa_results_data["source_info"] = f"Scraped: {source} - {url.split('?')[0]}"
                        st.sidebar.info("ABSA complete.")
                        
                        # Optional: Brief ABSA summary in chatbot
                        positive_aspects = absa_df_results[absa_df_results['sentiment_llm'] == 'positive']['aspect_term_llm'].nunique()
                        negative_aspects = absa_df_results[absa_df_results['sentiment_llm'] == 'negative']['aspect_term_llm'].nunique()
                        st.session_state.conversation.append(('assistant', 
                            f"\n**ABSA Insights (from scraped data):**\n"
                            f"- Found {positive_aspects} unique positive aspects.\n"
                            f"- Found {negative_aspects} unique negative aspects.\n"
                            f"(Explore details in 'ABSA Results Explorer' tab.)"
                        ))
                    else:
                        st.sidebar.warning("ABSA failed or returned no results on scraped data.")
                        st.session_state.absa_results_data["results_df"] = None # Clear if failed
            
            st.rerun()

    # Chat display area
    st.title("AI Chatbot")
    st.markdown("Enter a URL, select analyses, scrape data, and then chat or explore results!")
    for role, msg in st.session_state.conversation:
        with st.chat_message(role):
            st.markdown(msg)

    if user_query := st.chat_input("Ask about the scraped content..."):
        st.session_state.conversation.append(('user', user_query))
        st.session_state.messages_history.append({'role': 'user', 'content': user_query})
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            if st.session_state.messages_history:
                for chunk in get_openai_streaming_response(st.session_state.messages_history):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            else:
                full_response = "Error: Chat history is empty."
                message_placeholder.markdown(full_response)

        st.session_state.conversation.append(('assistant', full_response))
        st.session_state.messages_history.append({'role': 'assistant', 'content': full_response})


def run_topic_modeling_interactive():
    st.sidebar.subheader("Topic Modeling Explorer")

    source_info = st.session_state.topic_model_data.get("source_info")
    if source_info:
        st.sidebar.info(f"Current data: {source_info}")
    else:
        st.sidebar.info("No data loaded yet for topic modeling.")

    uploaded_file_topic = st.sidebar.file_uploader(
        "Upload new CSV to analyze (overrides current)", type=["csv"], key="topic_file_uploader_interactive_key"
    )

    df_current_for_modeler = st.session_state.topic_model_data.get("df_display_for_topic_model")

    if uploaded_file_topic:
        try:
            df_uploaded = pd.read_csv(uploaded_file_topic)
            if not df_uploaded.empty:
                st.session_state.topic_model_data.update({
                    "df": df_uploaded, # Store original uploaded df
                    "df_display_for_topic_model": df_uploaded.copy(), # Use a copy for modeler
                    "lda_model": None, "dominant_topics_df": None, "corpus": None, "id2word": None, # Reset model parts
                    "text_col": None, # Will be re-detected or selected
                    "source_info": f"Uploaded CSV: {uploaded_file_topic.name}"
                })
                df_current_for_modeler = df_uploaded.copy() # Update local var for immediate use
                st.sidebar.success(f"Using uploaded CSV: {uploaded_file_topic.name}. Please configure and train.")
                st.rerun()
            else:
                st.sidebar.warning("Uploaded CSV is empty.")
        except Exception as e:
            st.sidebar.error(f"Error reading uploaded CSV: {e}")
            df_current_for_modeler = None

    if df_current_for_modeler is not None and not df_current_for_modeler.empty:
        text_col_options = df_current_for_modeler.columns.tolist()
        
        # Determine default text column
        session_text_col = st.session_state.topic_model_data.get("text_col")
        idx = 0
        if session_text_col and session_text_col in text_col_options:
            idx = text_col_options.index(session_text_col)
        elif text_col_options: # Auto-detect if not set or not valid
            sensible = [c for c in text_col_options if any(k in c.lower() for k in ['text','comment','review','content'])]
            if sensible: idx = text_col_options.index(sensible[0])

        selected_text_col = st.sidebar.selectbox(
            "Select Text Column:", text_col_options, index=idx, key="explorer_text_col_select"
        )

        # If text column selection changes, model needs retraining
        if st.session_state.topic_model_data.get("text_col") != selected_text_col:
            st.session_state.topic_model_data["text_col"] = selected_text_col
            st.session_state.topic_model_data["lda_model"] = None # Invalidate model
            st.info("Text column changed. Please Train/Update model.")


        doc_count = len(df_current_for_modeler)
        max_topics = max(2, min(15, doc_count // 2 if doc_count > 4 else doc_count if doc_count > 1 else 2))
        default_topics = min(3, max_topics) if max_topics > 1 else 2
        
        num_topics_interactive = st.sidebar.slider(
            "Number of Topics:", 2, max_topics, default_topics, key="explorer_num_topics_slider"
        )
        custom_stopwords_interactive_str = st.sidebar.text_area(
            "Custom Stopwords:",
            value="thanks,thank,please,hi,hello,ok,buy,subscribe,follow,click,subject,re,edu,com,www,http,html,product,item,review,video,channel,comment",
            key="explorer_stopwords_area"
        )

        # Prompt to train if model is not present for current data/params
        if st.session_state.topic_model_data.get("lda_model") is None and df_current_for_modeler is not None:
            st.info("Data loaded/parameters set. Click 'Train/Update Topic Model' below.")

        if st.sidebar.button("Train/Update Topic Model", key="explorer_train_button"):
            custom_stopwords_list_interactive = [w.strip() for w in custom_stopwords_interactive_str.split(',') if w.strip()]
            
            # Use df_current_for_modeler and selected_text_col for the modeler instance
            modeler_explorer = get_topic_modeler_instance(
                df_current_for_modeler, selected_text_col, custom_stopwords_list_interactive
            )
            
            l_lda, l_corpus, l_id2word, l_dom_df, l_df_display = train_lda_and_get_details(
                modeler_explorer, num_topics_interactive,
                dict_no_below=1, dict_no_above=0.95 # Default filter params
            )

            st.session_state.topic_model_data.update({
                "lda_model": l_lda, "corpus": l_corpus, "id2word": l_id2word,
                "dominant_topics_df": l_dom_df,
                "df_display_for_topic_model": l_df_display, # The df used by modeler
                "text_col": selected_text_col # Crucial to store the selected text col
            })
            # Ensure source_info reflects that this was trained in explorer if it wasn't set by chatbot
            if not st.session_state.topic_model_data.get("source_info") or "Uploaded CSV" in st.session_state.topic_model_data.get("source_info", ""):
                 st.session_state.topic_model_data["source_info"] = "Configured in Topic Explorer"


            if l_lda:
                st.sidebar.success(f"Topic model updated with {num_topics_interactive} topics.")
            else:
                st.sidebar.error("Failed to train topic model.")
            st.rerun()

        # --- Display Logic (retrieves fresh from session state) ---
        lda_model_to_show = st.session_state.topic_model_data.get("lda_model")
        dominant_df_to_show = st.session_state.topic_model_data.get("dominant_topics_df")
        df_model_was_trained_on = st.session_state.topic_model_data.get("df_display_for_topic_model")
        text_col_model_was_trained_on = st.session_state.topic_model_data.get("text_col")

        if lda_model_to_show and dominant_df_to_show is not None and \
           df_model_was_trained_on is not None and text_col_model_was_trained_on:
            
            st.header(f"Interactive Topic Explorer: {st.session_state.topic_model_data.get('source_info', '')}")
            
            if not hasattr(lda_model_to_show, 'num_topics') or lda_model_to_show.num_topics == 0:
                st.warning("The trained LDA model appears to have no topics or is invalid.")
            else:
                topic_summaries = {}
                try:
                    for i in range(lda_model_to_show.num_topics):
                        words = [word for word, prob in lda_model_to_show.show_topic(i, topn=5)]
                        topic_summaries[f"Topic {i+1}: {', '.join(words)}"] = i # Use Topic i+1 for display
                except Exception as e:
                    st.error(f"Error generating topic summaries: {e}")
                    return


                if not topic_summaries:
                    st.warning("No topic summaries could be generated. Check model output.")
                else:
                    selected_topic_summary_key = st.sidebar.selectbox(
                        "Select Topic to Explore:", options=list(topic_summaries.keys()),
                        key="explorer_topic_select_key"
                    )
                    selected_topic_id = topic_summaries[selected_topic_summary_key]

                    st.subheader(f"Details for: {selected_topic_summary_key}")
                    col1, col2 = st.columns(2)

                    with col1: # Bar Chart
                        st.markdown("##### 📊 Top Words")
                        top_words_data = lda_model_to_show.show_topic(selected_topic_id, topn=15)
                        if top_words_data:
                            words_df = pd.DataFrame(top_words_data, columns=['Word', 'Probability'])
                            chart = alt.Chart(words_df).mark_bar().encode(
                                x=alt.X('Probability:Q', title=None),
                                y=alt.Y('Word:N', sort='-x', title=None),
                                tooltip=['Word', 'Probability']
                            ).properties(height=350, title=f"Top terms for Topic {selected_topic_id+1}")
                            st.altair_chart(chart, use_container_width=True)
                        else: st.write("No top words for this topic.")

                    with col2: # Word Cloud
                        st.markdown("##### ☁️ Word Cloud")
                        top_words_cloud_data = lda_model_to_show.show_topic(selected_topic_id, topn=30)
                        if top_words_cloud_data:
                            word_freq_dict = {word: prob for word, prob in top_words_cloud_data}
                            if word_freq_dict:
                                try:
                                    wc = WordCloud(width=400, height=300, background_color='white', colormap='viridis').generate_from_frequencies(word_freq_dict)
                                    fig, ax = plt.subplots()
                                    ax.imshow(wc, interpolation='bilinear')
                                    ax.axis('off')
                                    st.pyplot(fig)
                                except Exception as e_wc:
                                    st.warning(f"Word cloud error: {e_wc}")
                            else: st.write("Not enough distinct words for cloud.")
                        else: st.write("No words for cloud.")
                    
                    st.markdown(f"##### 📜 Representative Documents for Topic {selected_topic_id+1}")
                    # Filter dominant_df_to_show for selected_topic_id
                    rep_docs = dominant_df_to_show[dominant_df_to_show['Dominant_Topic'] == selected_topic_id].sort_values(by='Probability', ascending=False)
                    if not rep_docs.empty:
                        num_rep_docs_slider = st.slider("Number of representative documents:", 1, min(10, len(rep_docs)), 3, key=f"rep_docs_slider_{selected_topic_id}")
                        for _, row in rep_docs.head(num_rep_docs_slider).iterrows():
                            doc_original_id = row['Document_ID']
                            if 0 <= doc_original_id < len(df_model_was_trained_on):
                                original_text = df_model_was_trained_on[text_col_model_was_trained_on].iloc[doc_original_id]
                                with st.expander(f"Doc ID {doc_original_id} (Original) | Prob: {row['Probability']:.3f} | Status: {row.get('Status', 'N/A')}"):
                                    st.write(original_text)
                            else:
                                st.warning(f"Original document ID {doc_original_id} out of bounds for display.")
                    else:
                        st.write(f"No documents primarily represent Topic {selected_topic_id+1}.")
        
        elif df_current_for_modeler is not None and not lda_model_to_show : # Data loaded, but no model yet trained/active
            st.info("👈 Please configure parameters in the sidebar and click 'Train/Update Topic Model'.")

    else: # No df_current_for_modeler (no upload, no data from chatbot)
        st.info("👈 Upload a CSV or scrape data in the Chatbot tab to perform topic modeling.")


def run_absa_results_explorer():
    st.title("🔍 ABSA Results Explorer")
    st.sidebar.subheader("ABSA Data Source")

    absa_data = st.session_state.absa_results_data.get("results_df")
    source_info = st.session_state.absa_results_data.get("source_info")

    if source_info:
        st.sidebar.info(f"Displaying ABSA results for: {source_info}")
    else:
        st.sidebar.info("No ABSA data currently loaded. Scrape data with ABSA enabled in the Chatbot tab.")

    if absa_data is not None and not absa_data.empty:
        st.markdown("### Aspect-Based Sentiment Analysis Results")
        
        # Filter out rows where aspect is N/A (i.e., no aspects found for that review/segment)
        valid_absa_data = absa_data[absa_data['aspect_term_llm'] != "N/A"].copy() # Use .copy() to avoid SettingWithCopyWarning

        if valid_absa_data.empty:
            st.warning("No specific aspects were extracted from the text. The LLM found no aspects to analyze.")
            st.markdown("Full data (including 'N/A' aspects):")
            st.dataframe(absa_data, use_container_width=True)
            return

        # --- Overall Sentiment Distribution for Aspects ---
        st.markdown("#### Sentiment Distribution of Extracted Aspects")
        if 'sentiment_llm' in valid_absa_data.columns:
            sentiment_counts = valid_absa_data['sentiment_llm'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            chart_sentiment_dist = alt.Chart(sentiment_counts).mark_bar().encode(
                x='Sentiment:N',
                y='Count:Q',
                color='Sentiment:N',
                tooltip=['Sentiment', 'Count']
            ).properties(title="Overall Aspect Sentiment Counts")
            st.altair_chart(chart_sentiment_dist, use_container_width=True)
        
        # --- Top Aspects ---
        st.markdown("#### Top Mentioned Aspects")
        if 'aspect_term_llm' in valid_absa_data.columns:
            top_n_aspects = st.slider("Number of top aspects to show:", 3, 20, 10, key="absa_top_n_slider")
            aspect_counts = valid_absa_data['aspect_term_llm'].value_counts().nlargest(top_n_aspects).reset_index()
            aspect_counts.columns = ['Aspect Term', 'Mentions']
            chart_top_aspects = alt.Chart(aspect_counts).mark_bar().encode(
                x=alt.X('Mentions:Q'),
                y=alt.Y('Aspect Term:N', sort='-x'), # Sort by mentions
                tooltip=['Aspect Term', 'Mentions']
            ).properties(title=f"Top {top_n_aspects} Mentioned Aspects")
            st.altair_chart(chart_top_aspects, use_container_width=True)

        # --- Detailed View with Filters ---
        st.markdown("#### Detailed Aspect Sentiments")
        
        # Filters
        filter_col1, filter_col2 = st.columns(2)
        unique_aspects = sorted(valid_absa_data['aspect_term_llm'].unique().tolist())
        selected_aspects = filter_col1.multiselect("Filter by Aspect Term:", unique_aspects, default=[], key="absa_aspect_filter")
        
        unique_sentiments = sorted(valid_absa_data['sentiment_llm'].unique().tolist())
        selected_sentiments = filter_col2.multiselect("Filter by Sentiment:", unique_sentiments, default=[], key="absa_sentiment_filter")

        filtered_df = valid_absa_data.copy() # Start with valid data
        if selected_aspects:
            filtered_df = filtered_df[filtered_df['aspect_term_llm'].isin(selected_aspects)]
        if selected_sentiments:
            filtered_df = filtered_df[filtered_df['sentiment_llm'].isin(selected_sentiments)]
        
        st.dataframe(
            filtered_df[['original_review_id', 'aspect_term_llm', 'sentiment_llm', 'processed_segment']], 
            use_container_width=True,
            height=400
        )
        
        # Show original review text for selected rows if needed (more advanced)
        # For instance, if a user clicks on a row in the dataframe.

        st.markdown("---")
        st.markdown("### Raw ABSA Output (including 'N/A' aspects)")
        st.dataframe(absa_data, use_container_width=True)

    else:
        st.info("No ABSA results to display. Please run ABSA on scraped data via the Chatbot tab.")

if __name__ == "__main__":
    main()