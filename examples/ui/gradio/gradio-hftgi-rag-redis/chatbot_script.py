# Import necessary libraries
import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.common.exceptions import TimeoutException
import random
from selenium.webdriver.support.ui import Select

# driver = webdriver.Firefox()
# driver.get("https://llm-flan-tgis-llm-demo.apps.ai-dev01.kni.syseng.devcluster.openshift.com/")
# driver.set_window_size(1084, 811)
timeout = 10

for user in range(10):
    driver = webdriver.Firefox()
    driver.get("https://llm-ui-tgis-llm-demo.apps.ai-dev01.kni.syseng.devcluster.openshift.com")
    driver.set_window_size(1084, 811)

    element_present = EC.presence_of_element_located((By.CSS_SELECTOR, "#component-0 .scroll-hide"))
    WebDriverWait(driver, timeout).until(element_present)

    # User enters the customer name
    customer_input = driver.find_element(By.CSS_SELECTOR, "#component-3 .scroll-hide")
    customer_input.clear()  # Clearing any previous input
    customer_input.send_keys(f"Apple") 

    dropdown = driver.find_element(By.CSS_SELECTOR, "#component-4 .secondary-wrap")
    dropdown.click()

    option = driver.find_element(By.XPATH, "(//ul/li)[2]")
    option.click()

    submit=driver.find_element(By.CSS_SELECTOR, "#component-6")
    submit.click()

    model_id_element = driver.find_element(By.ID, "model_id")
    model_id_text = model_id_element.text  # Get the text from the WebElement

    if "Model: google/flan-t5-xl" in model_id_text:
        label_list = [1, 2, 3]
    else:
        label_list = [4, 5]    
    random_num = random.choice(label_list)
    labelname=str(random_num)+'-radio-label'
    label_id="label[data-testid='"+labelname+"']"

    # label_id = "label[data-testid='2-radio-label']"
    WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.CSS_SELECTOR, label_id))).click()
    time.sleep(7)  # Adding a delay for better simulation of user interaction

# Close the browser after the loop completes
#driver.quit()
