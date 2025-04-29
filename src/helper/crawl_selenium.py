# crawl_selenium.py
from bs4 import BeautifulSoup
import requests
import re
import json
import time
import random
import pandas as pd

gc_collect = __import__('gc').collect
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from helper.handleCaptcha import solve_captcha

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36',
    'Accept-Language': 'vi,en-US;q=0.9,en;q=0.8',
    'accept-encoding': 'gzip, deflate, br, zstd',
    'referer': 'https://www.amazon.com/',
    'accept': '*/*'
}

def setup_driver(headless=False):
    """Setup and return a Chrome webdriver with appropriate options"""
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument(f"--user-agent={HEADERS['User-Agent']}")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_window_size(1920, 1080)
    return driver


def get_product_info(url):
    """Extract Amazon product info and reviews."""
    driver = setup_driver(headless=True)
    try:
        driver.get(url)
        time.sleep(random.uniform(3, 5))
        if solve_captcha(driver):
            time.sleep(3)

        product = {}
        # Title
        try:
            elem = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "productTitle")))
            product['title'] = elem.text.strip()
        except:
            product['title'] = None
        # Price
        try:
            price = driver.find_element(By.CSS_SELECTOR, ".a-offscreen").get_attribute("innerText")
            product['price'] = price
        except:
            product['price'] = None
        # Rating
        try:
            rating = driver.find_element(By.CSS_SELECTOR, ".a-icon-alt").get_attribute("innerText")
            product['rating'] = rating
        except:
            product['rating'] = None
        # Reviews
        product['reviews'] = []
        try:
            for rev in driver.find_elements(By.CSS_SELECTOR, "li.review"):
                r = {}
                try:
                    r['title'] = rev.find_element(By.CSS_SELECTOR, "a[data-hook='review-title']").text.strip()
                except:
                    r['title'] = None
                try:
                    r['text'] = rev.find_element(By.CSS_SELECTOR, "span[data-hook='review-body']").text.strip()
                except:
                    r['text'] = None
                try:
                    r['author'] = rev.find_element(By.CSS_SELECTOR, "span.a-profile-name").text.strip()
                except:
                    r['author'] = None
                try:
                    r['date'] = rev.find_element(By.CSS_SELECTOR, "span[data-hook='review-date']").text.strip()
                except:
                    r['date'] = None
                product['reviews'].append(r)
        except:
            pass
        return product
    finally:
        driver.quit()
        gc_collect()


def get_ebay_reviews(url, num_reviews=10):
    """Extract review texts from an eBay product review page."""
    driver = setup_driver(headless=True)
    all_reviews = []
    try:
        driver.get(url)
        time.sleep(random.uniform(2, 4))
        page = 1
        while len(all_reviews) < num_reviews:
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'p[itemprop="reviewBody"]'))
            )
            elems = driver.find_elements(By.CSS_SELECTOR, 'p[itemprop="reviewBody"]')
            for el in elems:
                text = el.text.strip()
                if text:
                    all_reviews.append({'title': None, 'text': text, 'author': None, 'date': None})
                    if len(all_reviews) >= num_reviews:
                        break
            try:
                next_btn = driver.find_element(By.CSS_SELECTOR, 'a[rel="next"]')
                link = next_btn.get_attribute('href')
                if not link:
                    break
                driver.get(link)
                time.sleep(random.uniform(2, 4))
                page += 1
            except:
                break
        return {'comments': all_reviews[:num_reviews]}
    finally:
        driver.quit()
        gc_collect()


def get_youtube_comments(url, num_comments=10):
    """Extract comments from a YouTube video page."""
    driver = setup_driver(headless=True)
    comments = []
    try:
        driver.get(url)
        time.sleep(5)
        # Scroll to comments section
        driver.execute_script("window.scrollTo(0, 600);")
        time.sleep(2)
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.ID, "content-text"))
        )
        last_height = driver.execute_script("return document.documentElement.scrollHeight")
        while len(comments) < num_comments:
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(2)
            elems = driver.find_elements(By.ID, "content-text")
            for el in elems:
                text = el.text.strip()
                if text and text not in comments:
                    comments.append(text)
                    if len(comments) >= num_comments:
                        break
            new_height = driver.execute_script("return document.documentElement.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        return {'comments': [{'text': c} for c in comments[:num_comments]]}
    finally:
        driver.quit()
        gc_collect()

