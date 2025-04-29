from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

# Setup
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--log-level=3")
driver = webdriver.Chrome(options=options)

video_url = "https://www.youtube.com/watch?v=nfnq__xvmMc"
driver.get(video_url)
time.sleep(5)

# Scroll to load comments section
driver.execute_script("window.scrollTo(0, 600);")
time.sleep(2)

# Wait for comments to load
WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.ID, "content-text"))
)

# Scroll until no new comments load
last_height = driver.execute_script("return document.documentElement.scrollHeight")
while True:
    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
    time.sleep(2)
    new_height = driver.execute_script("return document.documentElement.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

# Extract comments
comment_elements = driver.find_elements(By.ID, "content-text")
comments = [c.text for c in comment_elements if c.text.strip()]

# Export
df = pd.DataFrame(comments, columns=["Comment"])
df.drop_duplicates(subset="Comment", inplace=True)
df.to_csv("youtube_comments.csv", index=False, encoding="utf-8-sig")
print(f"✅ Extracted {len(df)} unique comments and saved to youtube_comments.csv")

driver.quit()
