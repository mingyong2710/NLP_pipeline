from amazoncaptcha import AmazonCaptcha
import time
from selenium.webdriver.common.by import By

def solve_captcha(driver):
    """
    Automatically solve Amazon captcha if present
    Returns: 
        bool: True if captcha was detected and solved, False otherwise
    """
    try:
        # Check if captcha is present
        if "captcha" in driver.page_source.lower() or "robot" in driver.page_source.lower():
            print("🔍 Captcha detected! Attempting to solve...")
            
            # Find the captcha image
            try:
                captcha_element = driver.find_element(By.TAG_NAME, "img")
                captcha_url = captcha_element.get_attribute('src')
                
                # Verify it's actually a captcha image URL
                if 'captcha' in captcha_url.lower():
                    # Solve the captcha
                    captcha = AmazonCaptcha.fromlink(captcha_url)
                    solution = captcha.solve()
                    
                    print(f"✓ Captcha solved: {solution}")
                    
                    # Enter the solution
                    input_field = driver.find_element(By.ID, "captchacharacters")
                    input_field.send_keys(solution)
                    
                    # Submit the form
                    submit_button = driver.find_element(By.CLASS_NAME, "a-button-text")
                    submit_button.click()
                    
                    # Wait for page to load
                    time.sleep(3)
                    return True
            except Exception as e:
                print(f"⚠️ Error solving captcha: {str(e)}")
        
        return False
    except Exception as e:
        print(f"⚠️ Error in captcha handling: {str(e)}")
        return False