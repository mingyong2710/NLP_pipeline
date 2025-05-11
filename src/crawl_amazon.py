from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import random # For random delays

# Pandas import is removed from here; the function returns a list.
# The calling script will handle DataFrame creation and CSV saving.

def get_amazon_reviews(product_url: str, num_target_reviews: int, run_headless: bool = False):
    """
    Scrapes reviews from an Amazon product page.

    Args:
        product_url (str): The URL of the Amazon product page.
        num_target_reviews (int): The desired number of reviews to collect.
        run_headless (bool): If True, runs Chrome in headless mode.
                             Defaults to False (visible browser) for easier debugging
                             and manual CAPTCHA intervention.

    Returns:
        list: A list of review text strings. Returns an empty list if no reviews
              are found or an error occurs.
    """
    print(f"Initiating Amazon review scraping for URL: {product_url}")
    print(f"Target number of reviews: {num_target_reviews}")
    print(f"Run headless: {run_headless}")

    # --- Setup Chrome Options ---
    options = Options()
    if run_headless:
        options.add_argument("--headless")
    options.add_argument("--disable-gpu") # Recommended for headless and sometimes for headed
    options.add_argument("--window-size=1920,1080") # Consistent window size
    # Using a common, realistic user agent
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36')
    options.add_argument("--log-level=0") # Suppress non-critical DevTools messages
    options.add_argument("--lang=en-US,en;q=0.9") # Request English page content
    # Attempt to make Selenium less detectable
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    driver = None
    all_reviews_collected = []
    MAX_REVIEW_PAGES_TO_CRAWL = 50 # Safety limit for pagination
    # REVIEWS_PER_PAGE = 10 # Amazon typically shows 10 reviews per page (not strictly used in logic but good to know)

    try:
        driver = webdriver.Chrome(options=options)
        # Further attempt to hide webdriver presence
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        print(f"Navigating to product page: {product_url}")
        driver.get(product_url)
        time.sleep(random.uniform(3, 6)) # Wait for initial page load and scripts

        # --- 1. Find and Click "See all reviews" link ---
        see_all_reviews_link_element = None
        review_link_selectors = [
            (By.XPATH, '//a[@data-hook="see-all-reviews-link-foot"]'),
            (By.CSS_SELECTOR, 'a[data-hook="see-all-reviews-link-foot"]'),
            (By.PARTIAL_LINK_TEXT, "See all reviews"),
            (By.CSS_SELECTOR, 'div[data-hook="reviews-medley-footer"] a'),
            (By.ID, "reviews-medley-footer") # Sometimes the whole footer is clickable or contains the link directly
        ]

        print("Looking for 'See all reviews' link...")
        for i, (selector_type, selector_value) in enumerate(review_link_selectors):
            try:
                candidate_element = WebDriverWait(driver, 7).until( # Shorter timeout for each attempt
                    EC.presence_of_element_located((selector_type, selector_value))
                )
                # If the selector is for the footer div, try to find an 'a' tag within it
                if selector_value == "reviews-medley-footer" and candidate_element.tag_name != 'a':
                    see_all_reviews_link_element = candidate_element.find_element(By.TAG_NAME, "a")
                else:
                    see_all_reviews_link_element = candidate_element
                
                if EC.element_to_be_clickable(see_all_reviews_link_element):
                    print(f"✅ Found clickable 'See all reviews' link/area with selector strategy (attempt {i+1}): {selector_value}")
                    break
                else:
                    see_all_reviews_link_element = None # Not clickable, reset
            except (TimeoutException, NoSuchElementException):
                print(f"ℹ️ 'See all reviews' link not found or not clickable with selector strategy (attempt {i+1}): {selector_value}")
            if see_all_reviews_link_element: # Found a clickable one
                break
        
        if not see_all_reviews_link_element:
            print("⚠️ Could not find the 'See all reviews' link. Will attempt to scrape reviews from current page.")
        else:
            try:
                driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", see_all_reviews_link_element)
                time.sleep(random.uniform(1, 2))
                # Use JavaScript click if direct click fails, can be more robust
                driver.execute_script("arguments[0].click();", see_all_reviews_link_element)
                print("Clicked 'See all reviews' link. Navigating to reviews page...")
                time.sleep(random.uniform(4, 7)) # Wait for reviews page to load
            except Exception as e_click:
                print(f"Error clicking 'See all reviews' link: {e_click}. Will try to scrape from current page.")

        # --- 2. Scrape Reviews from Review Pages ---
        current_page_num = 1
        while current_page_num <= MAX_REVIEW_PAGES_TO_CRAWL and len(all_reviews_collected) < num_target_reviews:
            print(f"📄 Scraping review page {current_page_num} (URL: {driver.current_url}). Target: {len(all_reviews_collected)}/{num_target_reviews} reviews.")
            
            # --- CAPTCHA Check and Manual Intervention ---
            page_content_lower = driver.page_source.lower()
            is_captcha_present = (
                "amazon.com/errors/validatecaptcha" in driver.current_url.lower() or
                "captcha" in page_content_lower or
                "api-js.datadome.co" in page_content_lower or
                " मानव " in page_content_lower or # Hindi for human, sometimes seen
                driver.find_elements(By.CSS_SELECTOR, 'iframe[title*="hCaptcha"]') or
                driver.find_elements(By.CSS_SELECTOR, 'iframe[title*="reCAPTCHA"]')
            )

            if is_captcha_present:
                print("🤖 POTENTIAL CAPTCHA DETECTED!")
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                captcha_screenshot_path = f"debug_captcha_detected_{timestamp}.png"
                try:
                    driver.save_screenshot(captcha_screenshot_path)
                    print(f"📸 Screenshot saved to {captcha_screenshot_path}")
                except Exception as e_cap_ss:
                    print(f"Could not save CAPTCHA screenshot: {e_cap_ss}")

                if not run_headless:
                    print("The browser window is visible. Please solve the CAPTCHA manually in the browser.")
                    try:
                        input("Press Enter in this console after you have solved the CAPTCHA in the browser...")
                        print("Resuming script after manual CAPTCHA attempt...")
                        time.sleep(random.uniform(3,5)) # Give page a moment to fully load after CAPTCHA
                        continue # Retry scraping this page
                    except KeyboardInterrupt:
                        print("Script interrupted by user during CAPTCHA solving.")
                        break 
                else:
                    print("Running in headless mode, cannot solve CAPTCHA manually. Stopping scraper.")
                    break
            
            # --- Scrape Review Texts ---
            try:
                WebDriverWait(driver, 20).until( # Increased wait time
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div[data-hook="review"] span[data-hook="review-body"]'))
                )
                
                review_elements_on_page = driver.find_elements(By.CSS_SELECTOR, 'div[data-hook="review"] span[data-hook="review-body"]')
                
                if not review_elements_on_page and current_page_num == 1 and not all_reviews_collected:
                    print("No review elements ('span[data-hook=\"review-body\"]') found on the first reviews page. Product might have no reviews or page structure is different.")
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    driver.save_screenshot(f"debug_no_reviews_elements_found_{timestamp}.png")
                    break

                new_reviews_this_page = 0
                for review_el in review_elements_on_page:
                    if len(all_reviews_collected) >= num_target_reviews:
                        break 
                    review_text = review_el.text.strip()
                    if review_text:
                        all_reviews_collected.append(review_text)
                        new_reviews_this_page +=1
                
                print(f"✅ Found {new_reviews_this_page} reviews on this page. Total collected: {len(all_reviews_collected)}")

                if len(all_reviews_collected) >= num_target_reviews:
                    print(f"🎯 Reached target of {num_target_reviews} reviews.")
                    break
                
                if new_reviews_this_page == 0 and current_page_num > 1:
                    print("No new reviews found on this page, though review containers might have been present earlier. Assuming end.")
                    break


                # --- 3. Navigate to Next Review Page ---
                try:
                    # Try to find the 'li' with class 'a-last' and then the 'a' tag within it
                    next_page_li = WebDriverWait(driver,10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, 'ul.a-pagination li.a-last'))
                    )
                    if "a-disabled" in next_page_li.get_attribute("class"):
                        print("ℹ️ 'Next page' button (li.a-last) is disabled. Assuming end of reviews.")
                        break
                    
                    next_page_button = next_page_li.find_element(By.TAG_NAME, "a")
                    if not EC.element_to_be_clickable(next_page_button):
                         print("ℹ️ 'Next page' button found but not clickable. Assuming end of reviews.")
                         break

                    driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", next_page_button)
                    time.sleep(random.uniform(1,2))
                    driver.execute_script("arguments[0].click();", next_page_button) # JS click
                    print("Clicked 'Next page'. Navigating...")
                    time.sleep(random.uniform(4, 7)) # Wait for next page to load
                    current_page_num += 1
                except (TimeoutException, NoSuchElementException):
                    print("ℹ️ No 'Next page' button found or structure changed. Assuming end of reviews.")
                    break
                except Exception as e_next:
                    print(f"Error clicking next page: {e_next}")
                    break

            except TimeoutException:
                print(f"⚠️ TimeoutException: Review elements not found on page {current_page_num} after 20 seconds (or other wait issue).")
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                screenshot_path = f"debug_timeout_amazon_reviews_page_{current_page_num}_{timestamp}.png"
                html_path = f"debug_timeout_amazon_reviews_page_{current_page_num}_{timestamp}.html"
                try:
                    driver.save_screenshot(screenshot_path)
                    with open(html_path, "w", encoding="utf-8") as f_html:
                        f_html.write(driver.page_source)
                    print(f"📸 Saved screenshot to {screenshot_path} and page source to {html_path}.")
                except Exception as e_debug:
                    print(f"Could not save debug files for Amazon review timeout: {e_debug}")
                break 
            except Exception as e_page_scrape:
                print(f"An error occurred scraping page {current_page_num}: {e_page_scrape}")
                # Optionally save debug info here too
                break
                
    except Exception as e:
        print(f"A critical error occurred during Amazon review scraping: {e}")
        if driver: # If driver was initialized
            try:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                driver.save_screenshot(f"debug_amazon_critical_error_{timestamp}.png")
                with open(f"debug_amazon_critical_error_{timestamp}.html", "w", encoding="utf-8") as f_html:
                    f_html.write(driver.page_source)
                print(f"📸 Saved debug info for critical error.")
            except Exception as e_debug:
                print(f"Could not save debug info for critical error: {e_debug}")
    finally:
        if driver:
            driver.quit()
            print("Browser closed.")

    # Deduplicate (Amazon sometimes shows same review if navigation is odd) and trim
    final_reviews = list(dict.fromkeys(all_reviews_collected)) # Preserve order while deduplicating
    if len(final_reviews) > num_target_reviews:
        final_reviews = final_reviews[:num_target_reviews]

    print(f"\n🏁 Finished Amazon review scraping. Collected {len(final_reviews)} unique reviews.")
    return final_reviews