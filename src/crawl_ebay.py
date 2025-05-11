from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import json
import re # For extracting item ID

def _click_cookie_banner(driver):
    cookie_buttons_xpaths = [
        "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept all')]",
        "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'allow all')]",
        "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'agree')]",
        "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'got it')]",
        "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'ok')]",
        "//button[@id='gdpr-banner-accept']",
        "//button[contains(@class, 'cookie-accept')]"
    ]
    for xpath in cookie_buttons_xpaths:
        try:
            cookie_button = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.XPATH, xpath))
            )
            cookie_button.click()
            print("Clicked a cookie consent button.")
            time.sleep(1.5)
            return True
        except:
            continue
    print("No common cookie consent button found or clickable.")
    return False

def get_ebay_reviews(url: str, num_comments_to_crawl: int = 20, headless: bool = True) -> list:
    """
    Crawls an eBay feedback page for a target number of comments.

    Args:
        url (str): The eBay feedback URL to crawl.
        num_comments_to_crawl (int): The target number of feedback comments to retrieve.
                                     The function will attempt to get at least this many.
        headless (bool): Whether to run the browser in headless mode.

    Returns:
        list: A list of dictionaries, where each dictionary represents a feedback item.
              Returns an empty list if an error occurs or no feedback is found.
              The list may contain slightly more than num_comments_to_crawl due to
              how comments are loaded per scroll.
    """
    feedback_data = []
    collected_comments_count = 0
    max_consecutive_no_new_content_scrolls = 3 # Stop if X scrolls yield no new content

    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36")

    driver = None
    try:
        print("Setting up WebDriver...")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        print(f"WebDriver setup complete. Navigating to URL: {url}")

        driver.get(url)
        print("Page loading. Waiting for initial content and trying to handle cookie banner...")
        time.sleep(2)

        _click_cookie_banner(driver)

        print("Waiting a bit more for main content to load...")
        time.sleep(5)

        # --- Scrolling Logic based on num_comments_to_crawl ---
        last_height = driver.execute_script("return document.body.scrollHeight")
        scroll_attempt = 0
        consecutive_no_new_content = 0

        print(f"Starting to scroll to collect at least {num_comments_to_crawl} comments...")
        while collected_comments_count < num_comments_to_crawl:
            scroll_attempt += 1
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            print(f"Scrolled down (Attempt {scroll_attempt}). Waiting for content to load...")
            time.sleep(2.5) # Wait for new content

            # Temporarily parse to check count (this is a bit inefficient but direct)
            # A more optimized way might be to check if new elements appeared using Selenium,
            # but parsing and checking length is simpler to implement here.
            current_page_source = driver.page_source
            current_soup = BeautifulSoup(current_page_source, 'html.parser')
            current_feedback_cards = current_soup.find_all('li', class_='fdbk-container')
            
            # Update collected_comments_count based on ALL currently visible cards
            # This means we don't add to a running total, but re-evaluate after each scroll.
            # This is important as we will parse all cards at the end only once.
            newly_found_count = len(current_feedback_cards)

            print(f"  Currently {newly_found_count} feedback items visible on page.")

            if newly_found_count <= collected_comments_count and newly_found_count > 0 : # No new cards loaded, but some exist
                consecutive_no_new_content +=1
                print(f"  No new comments loaded on this scroll (consecutive: {consecutive_no_new_content}).")
            else:
                consecutive_no_new_content = 0 # Reset if new content or page was empty initially

            collected_comments_count = newly_found_count # Update with the total visible

            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height or consecutive_no_new_content >= max_consecutive_no_new_content_scrolls:
                if new_height == last_height:
                    print("Reached bottom of the page (height didn't change).")
                if consecutive_no_new_content >= max_consecutive_no_new_content_scrolls:
                    print(f"Stopping scroll: No new content after {max_consecutive_no_new_content_scrolls} consecutive scrolls.")
                break # Exit scroll loop
            last_height = new_height

            if scroll_attempt > num_comments_to_crawl + 10 : # Safety break: if we scroll too much
                print(f"Safety break: Exceeded {scroll_attempt} scroll attempts.")
                break
        # --- End Scrolling Logic ---

        print("Scrolling finished. Parsing final page content...")
        # Get the final page source after all scrolling
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        feedback_cards = soup.find_all('li', class_='fdbk-container')
        print(f"Found {len(feedback_cards)} feedback cards in total after scrolling.")

        if not feedback_cards:
            print("No feedback cards found after scrolling. The HTML structure might have changed, or content didn't load.")
            return []

        for card_index, card in enumerate(feedback_cards):
            if len(feedback_data) >= num_comments_to_crawl: # Stop processing if we've hit the target
                print(f"Collected {len(feedback_data)} comments, reaching target of {num_comments_to_crawl}.")
                break
            
            data = {}
            # (Sentiment, Comment, Buyer, Date, Item Info extraction logic remains the same as before)
            # 1. Sentiment
            try:
                sentiment_svg = card.select_one('div.fdbk-container__details__info__icon svg')
                if sentiment_svg:
                    aria_label = sentiment_svg.get('aria-label', '').lower()
                    data_test_id = sentiment_svg.get('data-test-id', '').lower()
                    if "positive" in aria_label or "positive" in data_test_id: data['sentiment'] = 'positive'
                    elif "neutral" in aria_label or "neutral" in data_test_id: data['sentiment'] = 'neutral'
                    elif "negative" in aria_label or "negative" in data_test_id: data['sentiment'] = 'negative'
                    else: data['sentiment'] = 'unknown_svg_attr'
                else: data['sentiment'] = 'unknown_no_svg'
            except Exception: data['sentiment'] = 'error'

            # 2. Feedback Comment
            try:
                comment_span = card.select_one('div.fdbk-container__details__comment span')
                data['comment'] = comment_span.get_text(strip=True) if comment_span else "N/A"
            except Exception: data['comment'] = 'error'

            # 3. Buyer
            try:
                buyer_span = card.select_one('div.fdbk-container__details__info__username span:first-child')
                data['buyer'] = buyer_span.get_text(strip=True) if buyer_span else "N/A"
            except Exception: data['buyer'] = 'error'

            # 4. Date
            try:
                date_span = card.select_one('span.fdbk-container__details__info__divide__time span')
                data['date'] = date_span.get_text(strip=True) if date_span else "N/A"
            except Exception: data['date'] = 'error'

            # 5. Item Title and Item ID
            try:
                item_span = card.select_one('div.fdbk-container__details__item-link span')
                if item_span:
                    item_full_text = item_span.get_text(strip=True)
                    data['item_title_full'] = item_full_text
                    match = re.search(r'\(#(\d+)\)$', item_full_text)
                    data['item_id'] = match.group(1) if match else "N/A"
                    data['item_url'] = f"https://www.ebay.com/itm/{data['item_id']}" if data['item_id'] != "N/A" else "N/A"
                else:
                    data.update({'item_title_full': "N/A", 'item_id': "N/A", 'item_url': "N/A"})
            except Exception:
                data.update({'item_title_full': 'error', 'item_id': "N/A", 'item_url': "N/A"})
            
            feedback_data.append(data)

    except Exception as e:
        print(f"An error occurred during crawling process: {e}")
        import traceback
        traceback.print_exc()
        # Return what we have collected so far in case of partial success before error
        return feedback_data[:num_comments_to_crawl] if feedback_data else []
    finally:
        if driver:
            print("Closing WebDriver.")
            driver.quit()
    
    # Ensure we don't return more than requested, even if more were parsed from the last scroll
    return feedback_data[:num_comments_to_crawl]