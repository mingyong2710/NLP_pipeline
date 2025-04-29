from bs4 import BeautifulSoup
import requests
import re
import json
import time
import random
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from helper.handleCaptcha import solve_captcha
import gc

HEADERS = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36',
    'Accept-Language': 'vi,en-US;q=0.9,en;q=0.8',
    'accept-encoding': 'gzip, deflate, br, zstd',
    'referer': 'https://www.amazon.com/',
    'accept': '*/*'
}

def setup_driver(headless=False):  # Changed default to False
    """Setup and return a Chrome webdriver with appropriate options"""
    chrome_options = Options()
    
    if headless:
        chrome_options.add_argument("--headless")  # Run in headless mode (no UI)
    
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(f"user-agent={HEADERS['User-Agent']}")
    
    # Add some preferences that make detection harder
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    # Use webdriver-manager to handle driver installation
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # Set window size to typical desktop
    driver.set_window_size(1920, 1080)
    return driver

def get_product_info(url):
    """Extract basic product information from an Amazon product page using Selenium"""
    driver = setup_driver(headless=True)
    try:
        driver.get(url)
        time.sleep(random.uniform(3, 5))  # Random delay to mimic human behavior
        
        # Check if we hit a CAPTCHA and try to solve it
        if solve_captcha(driver):
            print("🔄 Continuing after captcha solution...")
            time.sleep(3)  # Wait a bit for the page to fully load

        # Product dictionary to store all information
        product = {}
        
        # Get product title
        try:
            title_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "productTitle"))
            )
            product["title"] = title_element.text.strip()
        except:
            product["title"] = "Title not found"
        
        # Get product price
        try:
            price_element = driver.find_element(By.CSS_SELECTOR, ".a-offscreen")
            product["price"] = price_element.get_attribute("innerText")
        except:
            product["price"] = "Price not found"
        
        # Get product rating
        try:
            rating_element = driver.find_element(By.CSS_SELECTOR, ".a-icon-alt")
            product["rating"] = rating_element.get_attribute("innerText")
        except:
            product["rating"] = "Rating not found"
        
        # Get number of reviews
        try:
            review_count_element = driver.find_element(By.ID, "acrCustomerReviewText")
            product["review_count"] = review_count_element.text
        except:
            product["review_count"] = "Review count not found"
        
        # Get product description
        try:
            description_element = driver.find_element(By.ID, "feature-bullets")
            product["description"] = description_element.text
        except:
            product["description"] = "Description not found"

        # Get any information from table (if available) <table class="a-normal a-spacing-micro">
        try:
            product["table"] = {}  # Initialize the table dictionary
            table_element = driver.find_element(By.CSS_SELECTOR, ".a-normal.a-spacing-micro")
            table_rows = table_element.find_elements(By.TAG_NAME, "tr")
            for row in table_rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) == 2:
                    key = cells[0].text.strip()
                    value = cells[1].text.strip()
                    product["table"][key] = value
        except:
            product["table"] = {}  # Ensure table key exists even if there's an error
        
        # Get product images
        try:
            img_element = driver.find_element(By.ID, "landingImage")
            img_data = img_element.get_attribute('data-a-dynamic-image')
            if img_data:
                img_urls = json.loads(img_data)
                product["images"] = list(img_urls.keys())
            else:
                product["images"] = [img_element.get_attribute('src')]
        except:
            product["images"] = []

        # Get reviews <li id="" data-hook="review" class="review aok-relative">
        try:
            product["reviews"] = []  # Initialize the reviews list
            review_elements = driver.find_elements(By.CSS_SELECTOR, "li.review")
            print(f"Found {len(review_elements)} reviews")
            for review in review_elements:
                review_dict = {}
                # review_dict["title"] = review.find_element(By.CSS_SELECTOR, "a[data-hook='review-title']").text.strip()
                # review_dict["rating"] = review.find_element(By.CSS_SELECTOR, "i[data-hook='review-star-rating']").text.strip()
                # review_dict["text"] = review.find_element(By.CSS_SELECTOR, "span[data-hook='review-body']").text.strip()
                # review_dict["author"] = review.find_element(By.CSS_SELECTOR, "span.a-profile-name").text.strip()
                # review_dict["date"] = review.find_element(By.CSS_SELECTOR, "span[data-hook='review-date']").text.strip()
                # product["reviews"].append(review_dict)

                # For the review title
                try:
                    # Try first with anchor tag (original reviews)
                    try:
                        title_element = review.find_element(By.CSS_SELECTOR, "a[data-hook='review-title']")
                        review_dict["title"] = title_element.text.strip()
                    except:
                        # If not found, try with span (reviews from other countries)
                        title_element = review.find_element(By.CSS_SELECTOR, "span[data-hook='review-title']")
                        review_dict["title"] = title_element.text.strip()
                except:
                    review_dict["title"] = "No title"

                # For the review text
                try:
                    text_element = review.find_element(By.CSS_SELECTOR, "span[data-hook='review-body']")
                    review_dict["text"] = text_element.text.strip()
                except:
                    review_dict["text"] = "No review text"
                
                # For the author
                try:
                    author_element = review.find_element(By.CSS_SELECTOR, "span.a-profile-name")
                    review_dict["author"] = author_element.text.strip()
                except:
                    review_dict["author"] = "Anonymous"
                
                # For the review date
                try:
                    date_element = review.find_element(By.CSS_SELECTOR, "span[data-hook='review-date']")
                    review_dict["date"] = date_element.text.strip()
                except:
                    review_dict["date"] = "No date"

                product["reviews"].append(review_dict)
        except Exception as e:
            print(f"Error parsing reviews: {str(e)}")
            product["reviews"] = []  # Ensure reviews key exists even if there's an error
        
        return product
    except Exception as e:
        return {"error": str(e)}
    finally:
        driver.quit()  # Close the browser
        del driver  # Explicitly delete the driver object
        gc.collect()  # Force garbage collection

# def get_reviews(url, num_reviews=10):
#     """Fetch product reviews from Amazon product page using Selenium"""
#     driver = setup_driver()
#     try:
#         # First navigate to product page
#         driver.get(url)
#         time.sleep(random.uniform(2, 4))
        
#         # Check for robot check
#         if "robot" in driver.page_source.lower() or "captcha" in driver.page_source.lower():
#             return {"error": "Robot check detected on product page"}
        
#         # Try to find and click on "See all reviews" link
#         try:
#             reviews_link = driver.find_element(By.CSS_SELECTOR, "a[data-hook='see-all-reviews-link-foot']")
#             reviews_link.click()
#             time.sleep(random.uniform(3, 5))
#         except:
#             # If not found, try to construct the reviews URL
#             # Extract product ID from URL
#             product_id_match = re.search(r"/dp/([^/]+)", url)
#             if product_id_match:
#                 product_id = product_id_match.group(1)
#                 reviews_url = f"https://www.amazon.com/product-reviews/{product_id}"
#                 driver.get(reviews_url)
#                 time.sleep(random.uniform(3, 5))
        
#         reviews_list = []
#         page_num = 1
        
#         while len(reviews_list) < num_reviews and page_num <= 5:  # Limit to 5 pages
#             # Check for robot check on reviews page
#             if "robot" in driver.page_source.lower() or "captcha" in driver.page_source.lower():
#                 break
                
#             # Parse reviews on current page
#             soup = BeautifulSoup(driver.page_source, 'html.parser')
#             review_elements = soup.find_all("div", attrs={'data-hook': 'review'})
            
#             for review in review_elements:
#                 if len(reviews_list) >= num_reviews:
#                     break
                    
#                 review_dict = {}
                
#                 # Get review title
#                 title_element = review.find("a", attrs={'data-hook': 'review-title'})
#                 if not title_element:
#                     title_element = review.find("span", attrs={'data-hook': 'review-title'})
#                 review_dict["title"] = title_element.text.strip() if title_element else "No title"
                
#                 # Get review rating
#                 rating_element = review.find("i", attrs={'data-hook': 'review-star-rating'})
#                 if not rating_element:
#                     rating_element = review.find("i", attrs={'data-hook': 'cmps-review-star-rating'})
#                 review_dict["rating"] = rating_element.text.strip() if rating_element else "No rating"
                
#                 # Get review text
#                 body_element = review.find("span", attrs={'data-hook': 'review-body'})
#                 review_dict["text"] = body_element.text.strip() if body_element else "No review text"
                
#                 # Get review author
#                 author_element = review.find("span", attrs={'class': 'a-profile-name'})
#                 review_dict["author"] = author_element.text.strip() if author_element else "Anonymous"
                
#                 # Get review date
#                 date_element = review.find("span", attrs={'data-hook': 'review-date'})
#                 review_dict["date"] = date_element.text.strip() if date_element else "No date"
                
#                 reviews_list.append(review_dict)
            
#             # Go to next page if needed
#             if len(reviews_list) < num_reviews:
#                 try:
#                     next_link = driver.find_element(By.CSS_SELECTOR, ".a-pagination .a-last a")
#                     next_link.click()
#                     time.sleep(random.uniform(3, 5))
#                     page_num += 1
#                 except:
#                     break
        
#         return reviews_list
#     except Exception as e:
#         return {"error": str(e)}
#     finally:
#         driver.quit()

# if __name__ == "__main__":
#     # Test the functions
#     product_url = "https://www.amazon.com/SAMSUNG-Smartphone-Processor-ProScaler-Manufacturer/dp/B0DP3DFMHB/"
#     # product_url = "https://www.amazon.com/CHEF-iQ-Thermometer-Ultra-Thin-Monitoring/dp/B0C7JNJW2N/"
#     result = get_product_info(product_url)
#     print(result["reviews"][-1].get("text", "No review text"))