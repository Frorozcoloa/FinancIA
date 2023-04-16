from bs4 import BeautifulSoup
import requests
import json
import pandas as pd
import uuid
import click

def get_contet_news(bs:BeautifulSoup)->str:
    content_raw = bs.find("div", class_="html-content").find_all("p")
    content_list = list(map(lambda p: p.text, content_raw))
    content = " ".join(content_list)
    return content

def gets_news(url)->str:
    response = requests.get(url)
    if response.ok:
        try:
            bs = BeautifulSoup(response.content, "html.parser")
            contet = get_contet_news(bs)
            return contet
        except  Exception as e:
            print(e)
            print(url)
        
        return None

@click.command()
@click.option("--category", type=str)
@click.option("--name_csv", type=str)
def main(category, name_csv):
    """
    Aquí es donde se realiza el scrapper al la republica
    ejemplo: scraper_larepublica  --category macroeconomia --name_csv economia.csv
    """
    url = f"https://www.larepublica.co/api/pager/term?id={categorias[cat]}&take=1000&offset=0"
    response = requests.get(url)
    print(f"----{category}----")
    data = []
    if response.ok:
        values = json.loads(response.content)
        total = len(values)
        for idx,val in enumerate(values):
            url =  val["postUrl"]
            contet = gets_news(url)
            title = val["title"]
            new = {
                "news_id":str(uuid.uuid4()),
                "news_url_absolute": url,
                "news_init_date": val["create"],
                "news_final_date":val["create"],
                "news_title":title,
                "news_text_content":contet,
                "entailment":1,
                "category":category}
            data.append(new)
            print(f"van {idx+1}/{total}")

    df_data = pd.DataFrame(data)
    df_data.to_csv(f"dataset_republica_{name_csv}.csv")

if __name__ == '__main__':
    main()