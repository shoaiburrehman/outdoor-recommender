import cloudscraper

url = "https://www.bergfreunde.de/ausruestung/?search=hardshelljacken&n=0"
scraper = cloudscraper.create_scraper()
response = scraper.get(url)

with open("debug.html", "w", encoding="utf-8") as f:
    f.write(response.text)

print("Saved live HTML to debug.html! Open this file to check selectors.")