---
layout: splash
permalink: /sentiment-analysis/
title: "Sentiment Analysis using LLMs"
header:
  overlay_image: /assets/images/sentiment-analysis/sentiment-analysis.jpeg
excerpt: "Using Large Language Models to perform sentiment analysis of the news"
---

In this article we use large language models (LLMs) to compute the sentiment analysis of news titles. We use [mediastack](https://mediastack.com) to get the list of news and [OpenAI](https:/www.openai.com) for the sentiment analysis. Mediastack requires a key; the free service will suffice. We focus on news from Great Britain; of each news, we take into consideration the title and the description.


```python
with open('mediastack-key.txt', 'r') as f:
    mediastack_key = f.read()
```


```python
import http.client, urllib.parse
 
conn = http.client.HTTPConnection('api.mediastack.com')
 
items = []

for i in range(10):
    params = urllib.parse.urlencode({
        'access_key': mediastack_key,
        'countries': 'gb',
        'sort': 'published_desc',
        'sources': 'bbc',
        'limit': 100,
        'offset': 100 * i,
    })
    
    conn.request('GET', '/v1/news?{}'.format(params))
    
    res = conn.getresponse()
    data = res.read()

    import json
    news = json.loads(data.decode('utf-8'))
    assert news['pagination']['offset'] == 100 * i
    items += news['data']
```


```python
import pandas as pd

df = pd.DataFrame.from_dict(items)
```


```python
with open('open-ai-key.txt', 'r') as f:
    openai_api_key = f.read()
```

It is convenient to use frameworks like [langchain](https://www.langchain.com/) instead of rolling our own version. We ask for the sentiment, in a scale from -2 to 2, but also the topic, to be selected from a given list, and the location, to be selected as well from a list; in addition we request the reasoning that has been applied.


```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-3.5-turbo-1106', temperature=0.1, openai_api_key=openai_api_key) 

description = f"""
You are asked to perform sentiment analysis on the following news header from the BBC website, \
with the higher the number the more positive the sentiment
""".strip()

schema = {
    'properties': {
        'sentiment': {
            'type': 'integer', 
            'enum': [-2, -1, 0, 1, 2], 
            'description': description,
        },
        'topic': {
            'type': 'string',
            'enum': ['business', 'entertainement', 'politics', 'education', 'sports', 'technology', 'health'],
            'description': "You are asked to classify the topic to which the news header refers to",
        },
        'location': {
            'type': 'string',
            'enum': ['local', 'state', 'continent', 'world'],
            'description': "You are asked to classify the region of interest of the news header",
        },
        'reasoning':{
            'type':'string', 
            'description': '''Explain, in a concise way, why you gave that particular tag to the news. If you can't complete the request, say why.'''
        }
    },
    'required':['sentiment', 'topic', 'location', 'reasoning']
}

from langchain.chains import create_tagging_chain 
chain = create_tagging_chain(schema=schema, llm=llm)  


def tagger(row): 
    try: 
        text = row.title + '. ' + row.description
        sentiment = chain.invoke(text)
        return sentiment['text']['sentiment'], sentiment['text']['topic'], sentiment['text']['location'], sentiment['text']['reasoning']
    except Exception as e:
        print(text, e)
        return 'Failed request: ' + str(e) 
```


```python
tags = df.apply(tagger, result_type='expand', axis=1)
```


```python
tags
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2</td>
      <td>sports</td>
      <td>local</td>
      <td>The news headline is about a local sports even...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>sports</td>
      <td>local</td>
      <td>The news headline refers to a local sports eve...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2</td>
      <td>education</td>
      <td>local</td>
      <td>The news header mentions a local incident in F...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1</td>
      <td>entertainment</td>
      <td>local</td>
      <td>The news is related to the entertainment indus...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-2</td>
      <td>sports</td>
      <td>local</td>
      <td>The news refers to a local sports event in Bri...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>2</td>
      <td>politics</td>
      <td>world</td>
      <td>The news refers to Ukraine's president Zelensk...</td>
    </tr>
    <tr>
      <th>996</th>
      <td>-2</td>
      <td>health</td>
      <td>local</td>
      <td>The passage is about a local health issue, spe...</td>
    </tr>
    <tr>
      <th>997</th>
      <td>2</td>
      <td>politics</td>
      <td>local</td>
      <td>The news header refers to a local political ev...</td>
    </tr>
    <tr>
      <th>998</th>
      <td>0</td>
      <td>technology</td>
      <td>world</td>
      <td>The passage discusses the alteration of a phot...</td>
    </tr>
    <tr>
      <th>999</th>
      <td>-2</td>
      <td>politics</td>
      <td>local</td>
      <td>The passage contains political news, specifica...</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 4 columns</p>
</div>




```python
tags.columns = ['pred_sentiment', 'pred_topic', 'pred_location', 'pred_reason']
results = pd.concat([df, tags], axis=1)
results = results[['title', 'description', 'url', 'category', 'pred_sentiment', 'pred_category', 'pred_location', 'pred_reason']]
for column in ['category', 'pred_category']:
    results[column] = results[column].astype('category')
```


```python
df.to_pickle('./df.pickle')
tags.to_pickle('./tags.pickle')
results.to_pickle('./results.pickle')
```


```python
results.groupby('category', observed=True).pred_sentiment.mean().to_frame()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pred_sentiment</th>
    </tr>
    <tr>
      <th>category</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>business</th>
      <td>-0.354167</td>
    </tr>
    <tr>
      <th>entertainment</th>
      <td>0.460000</td>
    </tr>
    <tr>
      <th>general</th>
      <td>-0.412054</td>
    </tr>
    <tr>
      <th>health</th>
      <td>-0.600000</td>
    </tr>
    <tr>
      <th>politics</th>
      <td>-0.525424</td>
    </tr>
    <tr>
      <th>science</th>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>technology</th>
      <td>0.363636</td>
    </tr>
  </tbody>
</table>
</div>


Finally we plot the results. Not unsurprisingly, most news are either very negative or very positive, as they are 'catchier' than neutral (i.e., informative but not emotionally binding) news. In this sample of news there is a small skew towards negative sentiment. Given the small amount of text most of the news are classified as "general". We would need to provide the article to obtain more realistic results, yet it is impressible how much can be achieved with so little content!


```python
import seaborn as sns 
sns.histplot(results, x="pred_sentiment", hue="category", multiple="stack",
             bins=[-3, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5], shrink=0.8)
```




    <Axes: xlabel='pred_sentiment', ylabel='Count'>




    
![png](/assets/images/sentiment-analysis/sentiment-analysis-1.png)
    

