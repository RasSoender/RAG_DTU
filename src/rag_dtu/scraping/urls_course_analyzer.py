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


def scrape_href(driver, xpath, wait_time=10):
    try:
        element = WebDriverWait(driver, wait_time).until(EC.presence_of_element_located((By.XPATH, xpath)))
        return element.get_attribute("href")
    except Exception as e:
        return None

def saving_into_json(course_urls):
    file_path = "data/reference_data.json"
    with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)  # Load existing data
    
    data["course_analyzer_urls"] = course_urls

    # Save the updated JSON back to the file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"JSON file {file_path} updated successfully!")


def main():
    url = "https://dtucourseanalyzer.pythonanywhere.com/browse?msc=course_type"
    
    urls = []

    driver = setup_driver(headless=True)  # Set to False if you want to see the browser
    driver.get(url)
    time.sleep(2)

    # Refresh page
    driver.refresh()
    time.sleep(2)

    course_n = 797

    for i in range(2, course_n+1):
        link_xpath = f"//div[@class='container']//section//div[@class='col'][{i}]//div[@class='card h-100']/a" 
        url_course = scrape_href(driver, link_xpath)

        if i % 50 == 0:
            print(i)
        if url_course:
            urls.append(url_course)
   
    saving_into_json(urls)

    driver.quit()


if __name__ == "__main__":
    main()

