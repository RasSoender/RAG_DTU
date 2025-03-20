import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig, CacheMode
from typing import List
import json

def save_to_file(title, result):
    folder = "study_programmes"
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
    urls = ['http://sdb.dtu.dk/2024/136/2348', 'http://sdb.dtu.dk/2024/136/2377', 'http://sdb.dtu.dk/2024/136/2349', 
            'http://sdb.dtu.dk/2024/136/2379', 'http://sdb.dtu.dk/2024/136/2350', 'http://sdb.dtu.dk/2024/136/2375', 
            'http://sdb.dtu.dk/2024/136/2351', 'http://sdb.dtu.dk/2024/136/2352', 'http://sdb.dtu.dk/2024/136/2353',
            'http://sdb.dtu.dk/2024/136/2387', 'http://sdb.dtu.dk/2024/136/2347', 'http://sdb.dtu.dk/2024/136/2354', 
            'http://sdb.dtu.dk/2024/136/2356', 'http://sdb.dtu.dk/2024/136/2357', 'http://sdb.dtu.dk/2024/136/2359',
            'http://sdb.dtu.dk/2024/136/2360', 'http://sdb.dtu.dk/2024/136/2361', 'http://sdb.dtu.dk/2024/136/2371',
            'http://sdb.dtu.dk/2024/136/2362', 'http://sdb.dtu.dk/2024/136/2363', 'http://sdb.dtu.dk/2024/136/2372', 
            'http://sdb.dtu.dk/2024/136/2364', 'http://sdb.dtu.dk/2024/136/2408', 'http://sdb.dtu.dk/2024/136/2376',
            'http://sdb.dtu.dk/2024/136/2365', 'http://sdb.dtu.dk/2024/136/2358', 'http://sdb.dtu.dk/2024/136/2366',
            'http://sdb.dtu.dk/2024/136/2367', 'http://sdb.dtu.dk/2024/136/2368', 'http://sdb.dtu.dk/2024/136/2355', 
            'http://sdb.dtu.dk/2024/136/2369', 'http://sdb.dtu.dk/2024/136/2370', 'http://sdb.dtu.dk/2024/136/2380',
            'http://sdb.dtu.dk/2024/136/2373', 'http://sdb.dtu.dk/2024/136/2374']
    
    title = ['Applied Chemistry', 'Architectural Engineering', 'Autonomous Systems', 'Bioinformatics and Systems Biology', 'Biomaterial Engineering for Medicine', 
             'Biomedical Engineering', 'Biotechnology', 'Business Analytics', 'Chemical and Biochemical Engineering', 'Civil Engineering', 'Communication Technologies and System Design', 
             'Computer Science and Engineering', 'Design and Innovation', 'Earth and Space Physics and Engineering', 'Electrical Engineering', 'Engineering Acoustics', 'Engineering Light', 
             'Engineering Physics', 'Environmental Engineering', 'Food Technology', 'Human-oriented Artificial Intelligence', 'Industrial Engineering and Management', 'Materials and Manufacturing Engineering', 
             'Mathematical Modelling and Computation', 'Mechanical Engineering', 'Petroleum Engineering', 'Pharmaceutical Design and Engineering', 'Quantitative Biology and Disease Modelling', 'Quantitative Biology and Disease Modelling', 
             'Sustainable Energy', 'Sustainable Energy', 'Sustainable Fisheries and Aquaculture', 'Technology Entrepreneurship', 'Transport and Logistics', 'Wind Energy']
    
    asyncio.run(sequntial_crawl(urls, title))