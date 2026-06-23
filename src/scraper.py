import os
import time
# pyrefly: ignore [missing-import]
import cloudscraper
# pyrefly: ignore [missing-import]
from bs4 import BeautifulSoup
import pandas as pd
import re

OUTPUT_FILE = "data/products.csv"

# EXPANDED: 6 Core Commercial Pillars Matrix
CATEGORIES = {
    "jacken": "https://www.bergfreunde.de/outdoor-jacken/",
    "hosen": "https://www.bergfreunde.de/outdoor-hosen/",
    "schuhe": "https://www.bergfreunde.de/schuhe/",
    "ausruestung": "https://www.bergfreunde.de/ausruestung/",
    "rucksaecke": "https://www.bergfreunde.de/rucksaecke/",
    "klettergurte": "https://www.bergfreunde.de/klettergurte/"
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7"
}

def scrape_page(url):
    try:
        scraper = cloudscraper.create_scraper(delay=3)
        response = scraper.get(url, headers=HEADERS)
        if response.status_code != 200:
            print(f"Failed to fetch {url} — status {response.status_code}")
            return None
        return BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        print(f"Connection exception on {url}: {e}")
        return None

def clean_numerical_price(price_str):
    if not price_str or price_str == "N/A":
        return None
    try:
        match = re.search(r'([\d\.]+),(\d{2})', price_str)
        if match:
            clean_str = match.group(1).replace(".", "") + "." + match.group(2)
            return float(clean_str)
    except Exception:
        pass
    return None

def parse_products(soup, category_tag):
    products = []
    cards = soup.select("li.product-item") or soup.select("div.product-container") or soup.select("a.product-link")
    
    for card in cards:
        try:
            link_el = card if card.name == 'a' else card.select_one("a.product-link")
            if not link_el:
                continue
                
            aria_label = link_el.get("aria-label", "")
            if not aria_label:
                continue
                
            meta = {}
            for item in aria_label.split(";"):
                if ":" in item:
                    key, value = item.split(":", 1)
                    meta[key.strip().lower()] = value.strip()
            
            brand = meta.get("marke", "Unknown")
            name = meta.get("produktname", "Unknown")
            raw_price = meta.get("preis", "N/A")
            raw_old_price = meta.get("ursprünglicher preis", "N/A")
            
            price_clean = clean_numerical_price(raw_price)
            old_price_clean = clean_numerical_price(raw_old_price)
            
            rating_score = 0.0
            review_count = 0
            card_html = str(card)
            
            # Review Count Regex Extraction
            count_match = re.search(r'>\s*\(([\d\.]+)\)\s*<', card_html) or re.search(r'class="review-count"[^>]*>\s*\(([\d\.]+)\)', card_html)
            if count_match:
                review_count = int(re.sub(r'\D', '', count_match.group(1)))
                
            # Rating Score Regex Extraction Fallbacks
            width_match = re.search(r'width:\s*([\d\.]+)%', card_html)
            if width_match:
                pct = float(width_match.group(1))
                rating_score = round((pct / 100.0) * 5.0, 1)
            else:
                star_class_match = re.search(r'stars-(\d+)', card_html)
                if star_class_match:
                    rating_score = round(float(star_class_match.group(1)) / 10.0, 1)
                else:
                    text_score_match = re.search(r'([\d\.,]+)\s*von\s*5', card_html)
                    if text_score_match:
                        rating_score = float(text_score_match.group(1).replace(",", "."))
            
            if rating_score > 0.0 and review_count == 0:
                review_count = 1
                
            discount_pct = 0.0
            if old_price_clean and price_clean and old_price_clean > price_clean:
                discount_pct = round(((old_price_clean - price_clean) / old_price_clean) * 100, 2)
            
            img_el = link_el.select_one("img.product-image") or card.select_one("img")
            img = ""
            if img_el:
                img = img_el.get("src") or img_el.get("data-src") or ""
            
            href = link_el.get("href", "")
            link = href if href.startswith("http") else f"https://www.bergfreunde.de{href}"
            
            products.append({
                "name": name,
                "brand": brand,
                "price_raw": raw_price,
                "price_clean": price_clean,
                "old_price_clean": old_price_clean if old_price_clean else price_clean,
                "discount_percentage": discount_pct,
                "rating_score": rating_score,
                "review_count": review_count,
                "category": category_tag,
                "image": img,
                "url": link,
            })
        except Exception:
            continue
            
    return products

def run_scraper(max_pages_per_category=6):
    all_products = []
    
    for category_name, base_url in CATEGORIES.items():
        print(f"\n⚡ Ingesting Category Domain Target: [{category_name.upper()}]")
        
        for page_num in range(1, max_pages_per_category + 1):
            if page_num == 1:
                url = base_url
            else:
                url = f"{base_url}{page_num}/"
            
            print(f"  Scraping Page {page_num} -> {url}")
            soup = scrape_page(url)
            if soup is None:
                continue
                
            products = parse_products(soup, category_tag=category_name)
            print(f"  Captured {len(products)} item blocks.")
            if not products:
                break
                
            all_products.extend(products)
            time.sleep(3)
            
    if not all_products:
        print("\n❌ Error: Dataset is completely empty.")
        return pd.DataFrame()
        
    df = pd.DataFrame(all_products)
    
    # Clean-up Phase
    initial_rows = len(df)
    df.drop_duplicates(subset=['url'], keep='first', inplace=True)
    df = df[df['price_clean'].notna()]
    
    print(f"\n--- Data Quality Scrub Complete ---")
    print(f"Initial raw elements collected: {initial_rows}")
    print(f"Clean unique elements available: {len(df)}")
    
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"🎉 Success! 6-Pillar High-Volume Dataset written to {OUTPUT_FILE}")
    return df

if __name__ == "__main__":
    # 6 Categories * 6 Pages = 36 total pages scraped safely.
    run_scraper(max_pages_per_category=6)