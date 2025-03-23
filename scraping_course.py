from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import json
import concurrent.futures
import os

def setup_driver(headless=True):
    """Set up and return a Selenium WebDriver."""
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-images")  # Skip loading images
    
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)

def click_button(driver, wait_time=5):
    try:
        button = WebDriverWait(driver, wait_time).until(EC.element_to_be_clickable((By.XPATH, "//div[@class='pull-right']//span//a")))
        button.click()
        return True
    except Exception as e:
        print(f"Error clicking button: {e}")
        return False

def scrape_text(driver, xpath, wait_time=5):
    """Retrieve text from an element given its XPath."""
    try:
        element = WebDriverWait(driver, wait_time).until(EC.presence_of_element_located((By.XPATH, xpath)))
        return element.text.strip()
    except Exception as e:
        return None

def saving_into_json(scraped_info):
    # Ensure directory exists
    os.makedirs("data_courses", exist_ok=True)
    
    course_no = scraped_info["course title"][:5]
    with open(f"data_courses/{course_no}.json", "w", encoding="utf-8") as f:
        json.dump(scraped_info, f, ensure_ascii=False, indent=4)
        print(f"JSON file saved successfully for {course_no}!")
    return None

def scraping_elements_left(driver, j, scraped_info):
    # Get all rows at once to reduce DOM queries
    try:
        rows = driver.find_elements(By.XPATH, f"//div[@class='box information']//table[{j}]//tr")
        for row in rows:
            cols = row.find_elements(By.TAG_NAME, "td")
            if len(cols) >= 2:
                key = cols[0].text.strip()
                value = cols[1].text.strip()
                if key:  # Only add if key exists
                    scraped_info[key] = value
    except Exception as e:
        print(f"Error in scraping_elements_left: {e}")
    return scraped_info

def scraping_elements_right(driver, scraped_info):
    try:
        # Get all section titles at once
        all_keys = []
        keys_elements = driver.find_elements(By.XPATH, "//div[@class='col-md-6 col-sm-12 col-xs-12']//div[@class='box']//div[@class='bar']")
        for element in keys_elements:
            key = element.text.strip()
            if key:
                all_keys.append(key)
        
        # Get the full text content
        all_text = scrape_text(driver, "//div[@class='col-md-6 col-sm-12 col-xs-12']//div[@class='box']")
        
        if all_text and all_keys:
            # Split text by titles
            split_text_by_titles(all_text, all_keys, scraped_info)
    except Exception as e:
        print(f"Error in scraping_elements_right: {e}")
    
    return scraped_info

def split_text_by_titles(text, titles, scraped_info):
    if not text or not titles:
        return scraped_info
        
    lines = text.split("\n")  # Split text into lines
    
    title_n = len(titles)
    title_index = 1
    current_title = titles[0]
    next_title = titles[1] if title_index < title_n else None
    current_content = []

    for line in lines:
        line = line.strip()
        if next_title and line == next_title:
            scraped_info[current_title] = "\n".join(current_content).strip()
            
            current_title = next_title
            current_content = []
            title_index += 1
            next_title = titles[title_index] if title_index < title_n else None
        else:
            current_content.append(line)

    # Save the last section
    if current_title:
        scraped_info[current_title] = "\n".join(current_content).strip()

    return scraped_info

def process_url(url, index):
    try:
        print(f"Processing URL {index}")
        driver = setup_driver(headless=True)
        
        # Set timeout to avoid hanging on problematic pages
        driver.set_page_load_timeout(30)
        
        try:
            driver.get(url)
        except Exception as e:
            print(f"Error loading URL {url}: {e}")
            driver.quit()
            return False
            
        # Click the button to get English version of page
        if not click_button(driver):
            driver.quit()
            return False
            
        # Refreshing the page
        driver.refresh()
        time.sleep(1)  # Reduced sleep time
        
        scraped_info = dict()
        scraped_info["course title"] = scrape_text(driver, "//h2")
        
        if not scraped_info["course title"]:
            print(f"Failed to get course title for URL {url}")
            driver.quit()
            return False
            
        # Scrape left elements
        for j in range(1, 4):
            scraped_info = scraping_elements_left(driver, j, scraped_info)
        
        # Scrape right elements
        scraped_info = scraping_elements_right(driver, scraped_info)
        
        # Save results
        saving_into_json(scraped_info)
        
        driver.quit()
        return True
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return False

def main():
    with open("reference_data.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    # Extract the arrays
    course_urls = data.get("course_urls", [])
    
    start_index = 0
    urls_to_process = course_urls[start_index:]
    
    # Process URLs in parallel (adjust max_workers as needed)
    max_workers = 4  # Limit concurrent browsers to avoid memory issues
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks and keep track of them
        future_to_url = {executor.submit(process_url, url, i+start_index): (url, i+start_index) 
                         for i, url in enumerate(urls_to_process)}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_url):
            url, index = future_to_url[future]
            try:
                success = future.result()
                if success:
                    print(f"Successfully processed URL {index}")
                else:
                    print(f"Failed to process URL {index}")
            except Exception as e:
                print(f"URL {index} generated an exception: {e}")

if __name__ == "__main__":
    main()