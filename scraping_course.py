from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in the background (remove if you want to see the browser)
chrome_options.add_argument("--disable-blink-features=AutomationControlled")

# Start the WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# Open the webpage
url = "https://kurser.dtu.dk/course/47301"  # Update to the correct URL
driver.get(url)

time.sleep(2)  # Adjust as needed

button = driver.find_element(By.XPATH, "//div[@class='pull-right']//span//a")
button.click()

time.sleep(2)
driver.refresh() 

# Wait for the <h2> element to load
try:

    h2_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "//h2"))
    )

    course_title = h2_element.text
    print("Course Title:", course_title)

    label_element = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.XPATH, "//div[@class='box information']//table//tr//td/label"))
    )
    
    # Get the text of the label element
    label_text = label_element.text
    print("Label Text:", label_text)  

except Exception as e:
    print("Error:", e)

# Close the driver
driver.quit()
