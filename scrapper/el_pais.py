from scraper_utils import scrape_urls
import random
import requests
import time
import click

def generate_elpais_urls(query, page_nums):
    """
    Genera los URLS a scrapear de la revista de elpais. Este scraper funciona
    a partir de un query y una cantidad de paginas disponibles a esa query. Se recomienda hacer la query
    a partir de los resultados que lanza al utilizarlo en el search de la pagina
    """
    urls = []
    for i in range(1,int(page_nums)):
        query_str = "%22q%22:%22{}%22,%22page%22:{},%22limit%22:500,%22language%22:%22es%22".format(query,i)
        url = "https://elpais.com/pf/api/v3/content/fetch/enp-search-results?query={" + query_str + "}&_website=el-pais"
        urls.append(url)

    article_urls = []
    for _url in urls:
        try:
            req = requests.get(_url)
            time.sleep(random.uniform(0.1,0.25))
            req_json = req.json()
            print(len(req_json["articles"]))
            print(_url)
            for article_url in req_json["articles"]:
                article_urls.append(article_url["url"])
        except:
            print(f"URL: {_url} could not be scraped")
            continue

    print(f"Amount of article urls obtained: {len(article_urls)}")
    return article_urls


@click.command()
@click.option("--category", type=str)
@click.option("--query", type=str)
@click.option("--number", type=str)
@click.option("--csv_path", type=str)

#EXAMPLE scraper_elpais.py --category innovacion --query transformacion+digital+empresarial --number 30 --csv_path elpais_transdigiemp.csv
def main(category, query, number, csv_path):
    """ 
    Hay que poner este query para usar la función.
    Ejemplo: scraper_elpais.py --category innovacion --query transformacion+digital+empresarial --number 30 --csv_path elpais_transdigiemp.csv
    
    """
    webpage_urls = generate_elpais_urls(query,number)
    scraped_deduped = scrape_urls(webpage_urls, category)
    print(f'SCRAPED SEARCH QUERY {query}')
    scraped_deduped.to_csv(csv_path,index=False)



if __name__ == "__main__":
    main()