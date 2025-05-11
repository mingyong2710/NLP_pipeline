from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time

# Pandas import is removed from here; the function returns a list.
# The calling script will handle DataFrame creation and CSV saving.

def get_youtube_comments(video_url: str, num_target_comments: int):
    """
    Scrapes comments from a YouTube video.

    Args:
        video_url (str): The URL of the YouTube video.
        num_target_comments (int): The desired number of comments to collect.
                                   The scraper will try to load comments until this
                                   number is met or no more new comments load.

    Returns:
        list: A list of comment strings. Returns an empty list if no comments
              are found or an error occurs.
    """
    print(f"Initiating YouTube comment scraping for URL: {video_url}")
    print(f"Target number of comments: {num_target_comments}")

    # --- Setup Chrome Options ---
    options = Options()
    options.add_argument("--headless") # Consider running non-headless for easier debugging initially
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    # YouTube can be sensitive to user agents; a common one is good.
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36')
    options.add_argument("--log-level=0") # Suppress some console noise
    options.add_argument("--lang=en-US") # Request English page to make selectors more consistent
    options.add_argument("--mute-audio") # No need for sound

    driver = None
    all_comments_collected = []
    SCROLL_PAUSE_TIME = 2  # How long to wait for new comments to load after a scroll
    MAX_SCROLL_ATTEMPTS_WITHOUT_NEW_COMMENTS = 5 # Stop if no new comments after this many scrolls
    INITIAL_LOAD_WAIT_TIME = 15 # Time to wait for initial comments section to appear

    try:
        driver = webdriver.Chrome(options=options)
        print(f"Navigating to YouTube video: {video_url}")
        driver.get(video_url)
        time.sleep(3) # Allow initial page elements to settle

        # --- 1. Handle Potential Consent Pop-ups (common on YouTube) ---
        # YouTube uses different selectors for consent buttons depending on region/language.
        # This is a common one for "Accept all". You might need to adjust.
        consent_selectors = [
            (By.XPATH, '//button[@aria-label="Accept all"]'),
            (By.XPATH, '//button[@aria-label="Reject all"]'), # Alternative: reject non-essential
            (By.XPATH, '//yt-button-renderer[.//yt-formatted-string[contains(text(),"Accept all")]]'),
            (By.XPATH, '//tp-yt-paper-button[.//yt-formatted-string[contains(text(),"AGREE")]]'),
             # Add more selectors if you encounter different pop-ups
        ]
        
        consent_clicked = False
        for i, (selector_type, selector_value) in enumerate(consent_selectors):
            try:
                print(f"Looking for consent button (attempt {i+1})...")
                consent_button = WebDriverWait(driver, 5).until( # Shorter wait for each attempt
                    EC.element_to_be_clickable((selector_type, selector_value))
                )
                consent_button.click()
                print("✅ Clicked a consent button.")
                consent_clicked = True
                time.sleep(2) # Wait for overlay to disappear
                break # Exit loop once a button is clicked
            except TimeoutException:
                print(f"ℹ️ Consent button with selector '{selector_value}' not found.")
            except Exception as e_consent:
                print(f"⚠️ Error clicking consent button with selector '{selector_value}': {e_consent}")
        
        if not consent_clicked:
            print("ℹ️ No consent buttons found or clicked. Proceeding...")
            # Optionally save debug info if no consent button was handled,
            # as it might affect page loading.
            # timestamp = time.strftime("%Y%m%d-%H%M%S")
            # driver.save_screenshot(f"debug_no_consent_clicked_{timestamp}.png")

        # --- 2. Scroll to bring comments section into view ---
        print("Scrolling down to reveal comments section...")
        # Scroll a few times to ensure the comments section is triggered to load
        for _ in range(3):
            driver.execute_script("window.scrollBy(0, 800);") # Scroll down by 800px
            time.sleep(1)

        # --- 3. Wait for the first comment to appear (indicator that comments are loading) ---
        try:
            print(f"Waiting up to {INITIAL_LOAD_WAIT_TIME}s for the first comment to appear...")
            # YouTube comment text is often within 'yt-formatted-string#content-text'
            # The container for comments is usually #comments
            WebDriverWait(driver, INITIAL_LOAD_WAIT_TIME).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "ytd-comment-thread-renderer #content-text"))
            )
            print("✅ Initial comments section detected.")
        except TimeoutException:
            print(f"⚠️ TimeoutException: Comments section (#content-text within ytd-comment-thread-renderer) not detected after {INITIAL_LOAD_WAIT_TIME} seconds.")
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            screenshot_path = f"debug_timeout_initial_comments_{timestamp}.png"
            html_path = f"debug_timeout_initial_comments_{timestamp}.html"
            try:
                driver.save_screenshot(screenshot_path)
                with open(html_path, "w", encoding="utf-8") as f_html:
                    f_html.write(driver.page_source)
                print(f"📸 Saved screenshot to {screenshot_path} and page source to {html_path}.")
            except Exception as e_debug:
                print(f"Could not save debug files for initial comments timeout: {e_debug}")
            # If initial comments don't load, no point continuing
            if driver: driver.quit()
            return [] # Return empty list

        # --- 4. Scroll to load more comments until target is met or no new comments appear ---
        print(f"Scrolling to load comments. Target: {num_target_comments}")
        last_height = driver.execute_script("return document.documentElement.scrollHeight")
        scroll_attempts_no_new = 0
        
        # Extract initially visible comments
        comment_elements = driver.find_elements(By.CSS_SELECTOR, "ytd-comment-thread-renderer #content-text")
        for c_el in comment_elements:
            comment_text = c_el.text.strip()
            if comment_text and comment_text not in all_comments_collected:
                all_comments_collected.append(comment_text)
        print(f"  Initial comments found: {len(all_comments_collected)}")


        while len(all_comments_collected) < num_target_comments:
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(SCROLL_PAUSE_TIME) # Wait for new comments to load

            new_height = driver.execute_script("return document.documentElement.scrollHeight")
            
            current_comment_count = len(all_comments_collected)
            
            # Re-fetch comment elements after scroll
            comment_elements = driver.find_elements(By.CSS_SELECTOR, "ytd-comment-thread-renderer #content-text")
            
            new_comments_found_this_scroll = 0
            for c_el in comment_elements[current_comment_count:]: # Process only potentially new elements
                comment_text = c_el.text.strip()
                if comment_text and comment_text not in all_comments_collected: # Basic deduplication
                    all_comments_collected.append(comment_text)
                    new_comments_found_this_scroll +=1
            
            if new_comments_found_this_scroll > 0:
                print(f"  Loaded more comments. Total collected: {len(all_comments_collected)}/{num_target_comments}")
                scroll_attempts_no_new = 0 # Reset counter
            else:
                scroll_attempts_no_new += 1
                print(f"  No new unique comments found on this scroll ({scroll_attempts_no_new}/{MAX_SCROLL_ATTEMPTS_WITHOUT_NEW_COMMENTS}). Total: {len(all_comments_collected)}")

            if new_height == last_height and scroll_attempts_no_new >= 1: # If height didn't change AND no new text comments
                 print("  Page height hasn't changed and no new text comments after scroll. Assuming end of loadable comments.")
                 break
            if scroll_attempts_no_new >= MAX_SCROLL_ATTEMPTS_WITHOUT_NEW_COMMENTS:
                print(f"  No new comments after {MAX_SCROLL_ATTEMPTS_WITHOUT_NEW_COMMENTS} consecutive scrolls. Stopping.")
                break
            
            last_height = new_height
            
            if len(all_comments_collected) >= num_target_comments:
                print(f"🎯 Reached or exceeded target of {num_target_comments} comments.")
                break
    
    except Exception as e:
        print(f"A critical error occurred during YouTube comment scraping: {e}")
        if driver:
            try:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                driver.save_screenshot(f"debug_yt_critical_error_{timestamp}.png")
                with open(f"debug_yt_critical_error_{timestamp}.html", "w", encoding="utf-8") as f_html:
                    f_html.write(driver.page_source)
                print(f"📸 Saved debug info for critical error.")
            except Exception as e_debug:
                print(f"Could not save debug files for critical error: {e_debug}")
    finally:
        if driver:
            driver.quit()
            print("Browser closed.")
    
    # Trim if more comments were collected than targeted
    final_comments = list(dict.fromkeys(all_comments_collected)) # More robust deduplication
    if len(final_comments) > num_target_comments:
        final_comments = final_comments[:num_target_comments]
        
    print(f"\n🏁 Finished YouTube comment scraping. Collected {len(final_comments)} unique comments.")
    return final_comments
