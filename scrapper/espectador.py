from urllib import response
from bs4 import BeautifulSoup
from matplotlib.pyplot import axis
import requests
import json
import pandas as pd
import uuid
import os
import click

def generetes_url_espectador(response)->str:
    """Genera la URL de para poder extraer las noticas"""
    host = "https://www.elespectador.com/"
    canonical = response["canonical_url"]
    path = host + canonical
    return path

def get_contet_news(bs:BeautifulSoup)->str:
    """Obtenemos la información de la noticial"""
    content_raw = bs.find("section", class_="false").find_all("p")[:-1]
    content_list = list(map(lambda p: p.text, content_raw))
    content = " ".join(content_list)
    return content

def gets_news(url)->str:
    """Por cada articulo sacmos el texto, nombre y más información"""
    response = requests.get(url)
    if response.ok:
        try:
            bs = BeautifulSoup(response.content, "html.parser")
            title = bs.title.text
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
    Aquí es donde se realiza el scrapper al espectador
    ejemplo: scraper_especatador --category macroeconomia --name_csv economia_espectador.csv
    """
    offset = 100
    while True:
        try:
            url_tag = 'https://www.elespectador.com/pf/api/v3/content/fetch/general?query={"initOffset":'+str(offset) +',"match_phrase":{"taxonomy.tags.text":'+ f'"{category}"' +'},"page": 1,"size":100,"sourceInclude":"_id,additional_properties,canonical_url,type,subtype,description.basic,headlines.basic,subheadlines.basic,taxonomy.primary_section._id,taxonomy.primary_section.name,taxonomy.primary_section.path,taxonomy.sections.name,taxonomy.tags.text,taxonomy.tags.slug,first_publish_date,display_date,last_updated_date,promo_items.basic,promo_items.comercial,promo_items.comercial_movil,promo_items.jw_player,label,credits.by._id,credits.by.name,credits.by.additional_properties.original,credits.by.image.url,commentCount"}&d=610&_website=el-espectador'
            data = []
            response = requests.get(url_tag)
            if response.ok:
                news = json.loads(response.content)
                news = news["content_elements"]
                total_news = len(news)
                for idx,new in enumerate(news):
                    url = generetes_url_espectador(new)
                    date = new["first_publish_date"]
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