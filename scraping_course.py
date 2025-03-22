from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import json

def setup_driver(headless=True):
    """Set up and return a Selenium WebDriver."""
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")

    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)


def click_button(driver, wait_time=10):
    try:
        button = WebDriverWait(driver, wait_time).until(EC.element_to_be_clickable((By.XPATH, "//div[@class='pull-right']//span//a")))
        button.click()
        time.sleep(2)  # Adjust as needed
    except Exception as e:
        print(f"Error clicking button:", e)


def scrape_text(driver, xpath, wait_time=10):
    """Retrieve text from an element given its XPath."""
    try:
        element = WebDriverWait(driver, wait_time).until(EC.presence_of_element_located((By.XPATH, xpath)))
        return element.text.strip()
    except Exception as e:
        return None


def saving_into_json(scraped_info):
    course_no = scraped_info["course title"][:5]
    with open(f"data_courses/{course_no}.json", "w", encoding="utf-8") as f:
        json.dump(scraped_info, f, indent=4) # dump() will overwrite
        print("JSON file saved successfully!")
    return None

def scraping_elements_left(driver, j, scraped_info):
    i = 1
    while True: 
        key = scrape_text(driver, f"//div[@class='box information']//table[{j}]//tr[{i}]//td[1]")
        if not key:
            break
        value = scrape_text(driver, f"//div[@class='box information']//table[{j}]//tr[{i}]//td[2]")
        scraped_info[key] = value
        i += 1
    return scraped_info

def scraping_elements_right(driver, scraped_info):

    def split_text_by_titles(text, titles, scraped_info):
        lines = text.split("\n")  # Split text into lines
        
        title_n = len(titles)
        title_index = 1
        current_title = titles[0]
        next_title = titles[1]
        current_content = []

        for line in lines:
            line = line.strip()
            if line == next_title:
                scraped_info[current_title] = "\n".join(current_content).strip()
                
                current_title = next_title
                current_content = []
                title_index += 1
                if title_index < title_n:                  
                    next_title = titles[title_index]
            else:
                current_content.append(line)

        # Save the last section
        if current_title:
            scraped_info[current_title] = "\n".join(current_content).strip()

        return scraped_info 

    i = 1
    all_keys = []
    while True:
        key = scrape_text(driver, f"//div[@class='col-md-6 col-sm-12 col-xs-12']//div[@class='box']//div[@class='bar'][{i}]")
        if not key:
            break
        all_keys.append(key)
        i += 1

    all_text = scrape_text(driver, "//div[@class='col-md-6 col-sm-12 col-xs-12']//div[@class='box']")
    scraped_info = split_text_by_titles(all_text, all_keys, scraped_info)
    return scraped_info


def main():
    url = "https://kurser.dtu.dk/course/47301"

    driver = setup_driver(headless=True)  # Set to False if you want to see the browser
    driver.get(url)
    time.sleep(2)

    # Click the button to get english version of page
    click_button(driver)

    # Refresh page
    driver.refresh()
    time.sleep(2)

    scraped_info = dict()
    scraped_info["course title"] = scrape_text(driver, "//h2")

    for j in range(1, 4):
        scraped_info = scraping_elements_left(driver, j, scraped_info)

    scraped_info = scraping_elements_right(driver, scraped_info)

    saving_into_json(scraped_info)
    driver.quit()


if __name__ == "__main__":
    main()