import pandas as pd 
import newspaper as ns
from datetime import datetime
import uuid
import traceback



def scrape_urls(webpage_urls, category):
    """
    Utiliza los webpages para luego scraper y asignar una categoria a la noticia
    Args:
        webpage_urls (list): Urls a las cuales se le quieren hacer los scraping
        category (str): nombre de la categoria
    Returns:
        df: DataFrame con las noticas no duplicadadas
    """    
    scraped_info = {}

    for i,url in enumerate(webpage_urls):
        try:
            print(i)
            print('-'*50)
            print(url)
            article = ns.Article(url)
            article.download()
            article.parse()
            
            news_date = datetime.strftime(article.publish_date, "%Y-%m-%d")

            print(news_date)
            scraped_info[i] = {'news_id': uuid.uuid4(),
                            'news_url_absolute': url,
                            'news_init_date' : news_date,
                            'news_final_date' : news_date,
                            'news_title':article.title,
                            'news_text_content' : article.text,
                            'entailment' : 1,
                            'category':category
                                }
        except:
            print(traceback.print_exc())
    df = pd.DataFrame.from_records(scraped_info).T
    scraped_deduped = df.drop_duplicates(subset=['news_url_absolute'])
    print(f'Amount of news scraped: {len(scraped_deduped)}')
    return scraped_deduped