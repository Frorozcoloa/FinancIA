from urllib import response
from bs4 import BeautifulSoup
from matplotlib.pyplot import axis
import requests
import json
import pandas as pd
import uuid
import click
import os

def create_url(response)->str:
    """Crea la URL de para poder extraer las noticas"""
    host = "https://www.publimetro.co"
    canonical = response["websites"]["mwncolombia"]["website_url"]
    path = host + canonical
    return path

def get_contet_news(bs:BeautifulSoup)->str:
    content_raw = bs.find("article").find_all("p")[:-3]
    content_list = list(map(lambda p: p.text, content_raw))
    content = " ".join(content_list).replace("\xa0","")
    return content

def gets_news(url)->str:
    response = requests.get(url)
    if response.ok:
        try:
            bs = BeautifulSoup(response.content, "html.parser")
            title = bs.title.text.replace("NOTICIAS: ", "")
            contet = get_contet_news(bs)
            return contet, title
        except  Exception as e:
            print(e)
            print(url)
        
        return None


@click.command()
@click.option("--category", type=str)
@click.option("--name_csv", type=str)
def main(category, name_csv):
    """
    Aquí es donde se realiza el scrapper a publimetro
    ejemplo: scraper_publimetro  --category macroeconomia --name_csv economia.csv
    """
    offset = 100
    while True:
        try:
            url_tag = 'https://www.publimetro.co/pf/api/v3/content/fetch/story-feed-sections?query={"excludeSections":"","feature":"results-list","feedOffset":'+str(offset)+',"feedSize":100,"includeSections":"/'+category+'"}&filter={content_elements{_id,credits{by{_id,additional_properties{original{byline}},name,type,url}},description{basic},display_date,headlines{basic},label{basic{display,text,url}},owner{sponsored},promo_items{basic{resized_params{158x89,274x154},type,url},lead_art{promo_items{basic{resized_params{158x89,274x154},type,url}},type}},type,websites{mwncolombia{website_section{_id,name},website_url}}},count,next}&d=265&_website=mwncolombia'
            data = []
            response = requests.get(url_tag)
            if response.ok:
                news = json.loads(response.content)
                news = news["content_elements"]
                total_news = len(news)
                for idx,new in enumerate(news):
                    url = create_url(new)
                    date = new["display_date"]
                    val = gets_news(url)
                    if val:
                        new = {
                            "news_id":str(uuid.uuid4()),
                            "news_url_absolute": url,
                            "news_init_date": date,
                            "news_final_date":date,
                            "news_title":val[1],
                            "news_text_content":val[0],
                            "entailment":1,
                            "category":category}
                        data.append(new)
                    print(f"van {idx+1}/{total_news}")
                path_csv = f"{name_csv}.csv"
                if not os.path.exists(path_csv):
                    df_data = pd.DataFrame(data)
                    df_data = df_data.drop_duplicates()
                    df_data.to_csv(path_csv, index=False)
                else:
                    df = pd.read_csv(path_csv)
                    df_new_data = pd.DataFrame(data)
                    df_data =  pd.concat([df, df_new_data], axis=0)
                    df_data = df_data.drop_duplicates()
                    df_data.to_csv(path_csv, index=False)
                offset += 100
        except Exception as e:
            print(e)
            break
        
if __name__ == '__main__':
    main()