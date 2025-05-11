# ABSA.py (Modified `perform_absa_on_csv` and related parts)
import pandas as pd
from openai import OpenAI, RateLimitError, APIError
import os
import time
import nltk
from nltk.tokenize import sent_tokenize
import re
import json
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY environment variable not set.")
    # In a Streamlit app, you might want to handle this with st.error()
    # For now, we'll let it proceed and fail later if key is needed
    # exit(1) 

client = None
if OPENAI_API_KEY:
     client = OpenAI(api_key=OPENAI_API_KEY)
else:
     print("WARNING (ABSA.py): OpenAI client not initialized due to missing API key.")


OPENAI_MODEL = "gpt-4o-mini"

nltk_resource_to_check = ('punkt', 'tokenizers/punkt')
try:
    nltk.data.find(nltk_resource_to_check[1])
except LookupError:
    print(f"NLTK resource '{nltk_resource_to_check[0]}' not found. Attempting to download...")
    try:
        nltk.download(nltk_resource_to_check[0], quiet=False)
        nltk.data.find(nltk_resource_to_check[1])
        print(f"NLTK resource '{nltk_resource_to_check[0]}' downloaded successfully.")
    except Exception as e_download:
        print(f"Error downloading NLTK resource '{nltk_resource_to_check[0]}': {e_download}")
        # exit(1) # Avoid exiting in a library context

def preprocess_text_for_llm(text):
    if not isinstance(text, str): return ""
    text = text.strip()
    return text

def get_aspects_and_sentiments_from_llm(text_segment, max_retries=3, retry_delay=20):
    if not client: # Check if client was initialized
        print("  Error (ABSA): OpenAI client not available. Skipping LLM call.")
        return []
    # ... (rest of your get_aspects_and_sentiments_from_llm function remains the same)
    system_message = """You are an expert at analyzing product reviews.
Your task is to identify explicit aspect terms mentioned in the provided text segment and determine the sentiment (positive, negative, or neutral) expressed towards each aspect.

Provide your output as a JSON object. This JSON object should contain a single key, "results", which is a list of objects. Each object in the "results" list must have two keys: "aspect_term" (a string) and "sentiment" (a string: "positive", "negative", or "neutral").
If only one aspect is found, the "results" list will contain a single object.
If no aspects are explicitly mentioned or no clear sentiment is expressed for an aspect, return a JSON object with an empty "results" list: {"results": []}.
"""
    user_prompt = f"Here is the text segment to analyze:\n\n\"{text_segment}\"\n\nExtract aspects and their sentiments in the specified JSON format."
    current_retry = 0
    while current_retry < max_retries:
        try:
            response_format_arg = {"response_format": {"type": "json_object"}}
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt.strip()}
                ],
                temperature=0.0, max_tokens=512,
                response_format=response_format_arg["response_format"]
            )
            content = response.choices[0].message.content
            if not content: return []
            parsed_json_object = json.loads(content)
            if not isinstance(parsed_json_object, dict) or "results" not in parsed_json_object: return []
            parsed_output_list = parsed_json_object.get("results", [])
            if not isinstance(parsed_output_list, list): return []
            validated_output = []
            for item in parsed_output_list:
                if isinstance(item, dict) and "aspect_term" in item and "sentiment" in item:
                    sentiment_val = str(item["sentiment"]).lower()
                    if sentiment_val not in ["positive", "negative", "neutral"]: continue
                    item["sentiment"] = sentiment_val
                    item["aspect_term"] = str(item["aspect_term"])
                    validated_output.append(item)
            return validated_output
        except json.JSONDecodeError: print(f"  Warning: Failed to decode JSON. Raw: '{content}'"); return []
        except RateLimitError:
            current_retry += 1
            if current_retry >= max_retries: print(f"  Rate limit hit, max retries. Skipping."); return []
            print(f"  Rate limit, waiting {retry_delay}s (attempt {current_retry}/{max_retries})..."); time.sleep(retry_delay)
        except APIError as e:
            current_retry += 1
            if current_retry >= max_retries: print(f"  API Error, max retries: {e}. Skipping."); return []
            print(f"  API Error: {e}. Waiting {retry_delay}s (attempt {current_retry}/{max_retries})..."); time.sleep(retry_delay)
        except Exception as e: print(f"  Error calling OpenAI: {e}"); return []
    print(f"  Error: Max retries exceeded for segment '{text_segment[:50]}...'")
    return []


def perform_absa_on_dataframe(input_df, text_column_name, process_entire_review=False): # Changed function name and signature
    """
    Performs Aspect-Based Sentiment Analysis on a DataFrame.
    Args:
        input_df (pd.DataFrame): The DataFrame containing the text data.
        text_column_name (str): The name of the column with text.
        process_entire_review (bool): If True, process whole review; else, sentence by sentence.
    Returns:
        pd.DataFrame: A DataFrame with ABSA results, or an empty DataFrame on failure.
    """
    if not isinstance(input_df, pd.DataFrame) or input_df.empty:
        print("Error (ABSA): Input is not a valid or non-empty DataFrame.")
        return pd.DataFrame()

    if text_column_name not in input_df.columns:
        print(f"Error (ABSA): Column '{text_column_name}' not found in DataFrame.")
        print(f"Available columns: {input_df.columns.tolist()}")
        return pd.DataFrame()
    
    if not client: # Check if client was initialized (due to missing API key)
         print("Error (ABSA): OpenAI client not initialized. Cannot perform ABSA.")
         # Optionally, return a DataFrame with a status column indicating failure
         return pd.DataFrame([{
             "original_review_id": i,
             "original_review": input_df.iloc[i][text_column_name],
             "processed_segment": "N/A",
             "aspect_term_llm": "ERROR",
             "sentiment_llm": "OpenAI Client Not Initialized"
         } for i in range(len(input_df))])


    results_list = []
    total_rows = len(input_df)
    operation_mode = "Entire Review" if process_entire_review else "Sentence-Level"
    print(f"Starting {operation_mode} ABSA (OpenAI: {OPENAI_MODEL}) for {total_rows} texts...")

    for index, row in input_df.iterrows():
        raw_text = row[text_column_name]
        if pd.isna(raw_text) or not isinstance(raw_text, str) or not raw_text.strip():
            print(f"Skipping text {index + 1}/{total_rows} (missing/invalid).")
            continue

        cleaned_review_full = preprocess_text_for_llm(raw_text)
        if not cleaned_review_full:
            print(f"Skipping text {index + 1}/{total_rows} (empty after preprocess).")
            continue

        # print(f"\nProcessing text {index + 1}/{total_rows}: \"{cleaned_review_full[:100]}...\"") # Less verbose

        text_segments_to_process = []
        if process_entire_review:
            text_segments_to_process.append(cleaned_review_full)
        else:
            sentences = sent_tokenize(cleaned_review_full)
            text_segments_to_process.extend([s for s in sentences if len(s.split()) >= 3 and len(s) >= 10])

        if not text_segments_to_process and not process_entire_review :
             # print(f"  No suitable sentences for text {index + 1}/{total_rows}.") # Less verbose
             pass # Will be handled by review_had_any_aspect_output check

        review_had_any_aspect_output = False
        for seg_idx, segment_text in enumerate(text_segments_to_process):
            # if not process_entire_review:
            #     print(f"  Segment {seg_idx + 1}/{len(text_segments_to_process)}: \"{segment_text[:50]}...\"")

            aspect_sentiment_pairs = get_aspects_and_sentiments_from_llm(segment_text)

            if aspect_sentiment_pairs:
                review_had_any_aspect_output = True
                for item in aspect_sentiment_pairs:
                    # print(f"    LLM => Aspect: '{item['aspect_term']}', Sentiment: {item['sentiment']}") # Less verbose
                    results_list.append({
                        "original_review_id": index,
                        "original_review": raw_text, # Store original full text
                        "processed_segment": segment_text,
                        "aspect_term_llm": item['aspect_term'],
                        "sentiment_llm": item['sentiment']
                    })
            # elif not process_entire_review :
            #      print(f"    LLM no aspects in segment.") # Less verbose

        if not review_had_any_aspect_output: # If no aspects found in any segment or the whole review
            # print(f"  LLM found no aspects/sentiments in text {index+1}.") # Less verbose
            results_list.append({
                "original_review_id": index,
                "original_review": raw_text,
                "processed_segment": cleaned_review_full if process_entire_review else "N/A (No aspects in sentences)",
                "aspect_term_llm": "N/A",
                "sentiment_llm": "N/A"
            })
    
    if not results_list: return pd.DataFrame() # Return empty if nothing was processed
    return pd.DataFrame(results_list)


# --- Example Usage (can be kept for standalone testing of ABSA.py) ---
if __name__ == "__main__":
    dummy_csv_path = 'ebay_data_for_absa.csv' # Use a different name to avoid conflict
    if not os.path.exists(dummy_csv_path):
        print(f"Creating dummy CSV: {dummy_csv_path}")
        dummy_data = {
            'review_text': [ # Changed column name to be more descriptive
                "The screen is amazing and bright, but the speakers are a bit tinny.",
                "Battery life is excellent, works all day. Performance is also top-notch.",
                "I love this phone! The camera takes stunning photos.",
                "It's an okay product, nothing special to write home about.",
                "This is terrible, the software crashes constantly and customer support was unhelpful.",
            ], 'product_id': ['A1', 'B2', 'C3', 'D4', 'E5']}
        pd.DataFrame(dummy_data).to_csv(dummy_csv_path, index=False)

    if not OPENAI_API_KEY:
        print("Cannot run __main__ example without OPENAI_API_KEY set in .env file.")
    else:
        df_test = pd.read_csv(dummy_csv_path)
        absa_results_df = perform_absa_on_dataframe(df_test, text_column_name='review_text', process_entire_review=False)

        print("\n--- ABSA Results (from DataFrame input) ---")
        if not absa_results_df.empty:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1200, 'display.max_colwidth', 60):
                print(absa_results_df)
            absa_results_df.to_csv("absa_openai_from_df_output.csv", index=False)
            print("\nResults also saved to absa_openai_from_df_output.csv")
        else:
            print("No ABSA results generated.")