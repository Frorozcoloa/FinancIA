import pandas as pd
import newspaper as ns
from bs4 import BeautifulSoup
import httplib2
import uuid
import click
from scraper_utils import scrape_urls


def generate_search_urls_valora(query, page_amounts):
    """
    Genera los URLS a scrapear de la revista de valoraanalitik. Este scraper funciona
    a partir de un query y una cantidad de paginas disponibles a esa query. Se recomienda hacer la query
    a partir de los resultados que lanza al utilizarlo en el search de la pagina
    """
    http = httplib2.Http()
            

    page_urls = [f"https://www.valoraanalitik.com/page/{i}/?s={query}" for i in range(2,int(page_amounts))]
    print(page_urls[0:10])

    webpage_urls = []
    for i, url in enumerate(page_urls):
        status, response = http.request(url)
        soup = BeautifulSoup(response, 'html.parser')
        try:
            listings = soup.find_all("div", class_= "td_module_16 td_module_wrap td-animation-stack")
        except:
            print(f'Error scraping {url}')
        for listing in listings:
            webpage_urls.append(listing.a["href"])

    print(f"Amount of article urls obtained: {len(webpage_urls)}")
    return webpage_urls

"""
Recei
"""

@click.command()
@click.option("--category", type=str)
@click.option("--query", type=str)
@click.option("--number", type=str)
@click.option("--csv_path", type=str)
def main(query, category,number,csv_path):
    """
    Aquí es donde se realiza el scrapper al la semana
    ejemplo: scraper_valora  --category macroeconomia  --query macroeconomia+colombiana --number 50 --name_csv economia.csv
    """
    webpage_urls = generate_search_urls_valora(query,number)
    scraped_deduped = scrape_urls(webpage_urls, category)
    print(f'SCRAPED SEARCH QUERY {query}')
    scraped_deduped.to_csv(csv_path,index=False)

if __name__ == '__main__':
    main()