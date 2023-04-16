import httplib2
import json
import click 
from scraper_utils import scrape_urls


def generate_search_url_semana(query, max_step):
    """
    Genera los URLS a scrapear de la revista de semana. Este scraper funciona
    a partir de un query y una cantidad de paginas disponibles a esa query. Se recomienda hacer la query
    a partir de los resultados que lanza al utilizarlo en el search de la pagina
    """
    skeleton_url = "https://api.queryly.com/v4/search.aspx?queryly_key=06e63be824464567&initialized=1&=&query={}&endindex={}&batchsize=100&callback=&extendeddatafields=creator,imageresizer,promo_image&timezoneoffset=300"
    h = httplib2.Http(".cache")
    req_urls = [skeleton_url.format(query,offset) for offset in range(0,30*int(max_step),30)]

    links = []
    for i, url in enumerate(req_urls):
        resp, content = h.request(url, "GET")
        try:
            parsed_content = parse_json_req(content)
        except:
            print(f"Could not parse {url}")
        parsed_links = ["https://semana.com" + element["link"] for element in parsed_content]
        links +=parsed_links

    print(f"Amount of article urls obtained: {len(links)}")
    return links


def parse_json_req(content):
    content = content.decode("utf-8")
    content = content.replace('\\\\"',"")
    content = content.replace("\\\\'","")
    content = content.replace("\\\'","")
    index1 = content.index("parse")
    index2 = content.index("\');")
    content = content[index1+7:index2]
    content = json.loads(content)
    return content["items"]
    

@click.command()
@click.option("--category", type=str)
@click.option("--query", type=str)
@click.option("--number", type=str)
@click.option("--csv_path",type=str)
def main(category, query, number,csv_path):
    """
    Aquí es donde se realiza el scrapper al la semana
    ejemplo: scraper_semnana  --category macroeconomia  --query macroeconomia+colombiana --number 50 --name_csv economia.csv
    """
    webpage_urls = generate_search_url_semana(query,number)
    scraped_deduped = scrape_urls(webpage_urls, category)
    print(f'SCRAPED SEARCH QUERY {query}')
    scraped_deduped.to_csv(csv_path,index=False)

if __name__ == '__main__':
    main()