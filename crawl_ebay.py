from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

# Setup headless Chrome
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
driver = webdriver.Chrome(options=options)

# URL of the first review page
base_url = "https://www.ebay.com/urw/LASFIT-LED-License-Plate-Map-Light-Bulbs-168-T10-192-for-Nissan-Maxima-2009-2014/product-reviews/2286944286"

driver.get(base_url)
all_reviews = []
page = 1

while True:
    print(f"📄 Crawling page {page}...")
    
    # Wait for review section to appear
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'p[itemprop="reviewBody"]'))
        )
    except:
        print("⚠️ Review section not found.")
        break

    # Extract reviews
    review_elements = driver.find_elements(By.CSS_SELECTOR, 'p[itemprop="reviewBody"]')
    reviews = [r.text.strip() for r in review_elements if r.text.strip()]
    print(f"✅ Found {len(reviews)} reviews on page {page}")
    all_reviews.extend(reviews)

    # Check for next page
    try:
        next_button = driver.find_element(By.CSS_SELECTOR, 'a[rel="next"]')
        next_link = next_button.get_attribute("href")
        if not next_link:
            break
        driver.get(next_link)
        time.sleep(2)
        page += 1
    except:
        print("⚠️ No more pages.")
        break

# Save to CSV
df = pd.DataFrame(all_reviews, columns=["Review"])
df.to_csv("ebay_reviews.csv", index=False, encoding="utf-8-sig")
print(f"\n✅ Extracted {len(all_reviews)} reviews across {page} pages. Saved to ebay_reviews.csv")

driver.quit()
