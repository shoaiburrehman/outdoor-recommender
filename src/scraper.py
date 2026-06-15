import os
import time
import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
import re

# Target output location
OUTPUT_FILE = "data/products.csv"

# Configuration parameters
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7"
}

def scrape_page(url):
    """Fetch one page using cloudscraper to bypass Cloudflare protection layers."""
    try:
        # Instantiating the engine to automatically handle JavaScript challenges
        scraper = cloudscraper.create_scraper(delay=3)
        response = scraper.get(url, headers=HEADERS)
        
        if response.status_code != 200:
            print(f"Failed to fetch {url} — status {response.status_code}")
            return None
        return BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        print(f"Connection exception on {url}: {e}")
        return None

def parse_products(soup):
    products = []
    
    # Target the main product anchor tags directly
    cards = soup.select("a.product-link")
    
    for card in cards:
        try:
            # 1. Get the structured label text
            aria_label = card.get("aria-label", "")
            if not aria_label:
                continue
                
            # 2. Parse out the metadata using regex or string splitting
            # Example label: "Marke: Trollkids; Produktname: Kid's Fjell Dreamer...; Preis: ab 59,96 €"
            meta = {}
            for item in aria_label.split(";"):
                if ":" in item:
                    key, value = item.split(":", 1)
                    meta[key.strip().lower()] = value.strip()
            
            # Extract fields with safe fallbacks
            brand = meta.get("marke", "Unknown")
            name = meta.get("produktname", "Unknown")
            price = meta.get("preis", meta.get("ursprünglicher preis", "N/A"))
            
            # 3. Handle the Destination URL
            href = card.get("href", "")
            link = href if href.startswith("http") else f"https://www.bergfreunde.de{href}"
            
            # 4. Extract the Image source accurately
            img_el = card.select_one("img.product-image")
            img = ""
            if img_el:
                # Fallback pathing for responsive srcset configurations
                img = img_el.get("src") or img_el.get("data-src") or ""
            
            products.append({
                "name":  name,
                "brand": brand,
                "price": price,
                "image": img,
                "url":   link,
            })
        except Exception as e:
            # Uncomment for troubleshooting layout edge cases
            # print(f"Skipped a card due to: {e}")
            continue
            
    return products

def run_scraper(search_term, max_pages=3):
    """Iterate cleanly through search parameters using explicit product index boundaries."""
    all_products = []
    items_per_page = 48  # Default capacity rendered on their listings
    
    for page_num in range(1, max_pages + 1):
        # Calculate pagination offset index (n=0 for page 1, n=48 for page 2, etc.)
        offset = (page_num - 1) * items_per_page
        url = f"https://www.bergfreunde.de/ausruestung/?search={search_term}&n={offset}"
        
        print(f"Scraping page {page_num}: {url}")
        soup = scrape_page(url)
        
        if soup is None:
            print("  Skipping page due to collection error.")
            continue
            
        products = parse_products(soup)
        print(f"  Found {len(products)} products")
        all_products.extend(products)
        
        time.sleep(3)  # Respect structural rates to protect IP tracking flags
    
    if not all_products:
        print("\nParsing completely failed. Verify page elements manually in your local browser.")
        return pd.DataFrame()
        
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(all_products)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccess! Saved {len(df)} items directly into {OUTPUT_FILE}")
    return df

if __name__ == "__main__":
    run_scraper(search_term="hardshelljacken", max_pages=3)