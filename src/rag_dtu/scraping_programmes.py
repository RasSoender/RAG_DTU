import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig, CacheMode
from typing import List
import json

def save_to_file(title, result):
    folder = "data_study_programmes"
    filename = f"{folder}/{title.replace(' ', '_')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        result_dict = {
            "markdown": result.markdown
        }
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    print(f"Result saved to {filename}")


async def sequntial_crawl(urls, titles):
    browser_config = BrowserConfig(
        headless=True,
        verbose=True,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"]
    )

    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    crawler = AsyncWebCrawler(config=browser_config)
    print("starting Crawler")
    await crawler.start()

    try:
        for i, url in enumerate(urls): 
            print(f"Crawling {url}...")

            result = await crawler.arun(
                url=url,
                config=crawl_config
            )
            if result.success:
                print(result.url, "crawled OK!")
                save_to_file(titles[i], result)
            else:
                print("Failed", result.url,"-", result.error_message)

    except Exception as e: 
        print(f"Error: {e}")
    finally:
        print("Closing crawler")
        await crawler.close()


if __name__ == "__main__":
    with open("reference_data.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    # Extract the arrays
    study_programme_urls = data.get("study_programme_urls", [])
    study_programme_titles = data.get("study_programme_titles", [])
    
    asyncio.run(sequntial_crawl(study_programme_urls, study_programme_titles))