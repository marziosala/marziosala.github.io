---
layout: splash
permalink: /news-summary/
title: "What's in the News?"
header:
  overlay_image: /assets/images/news-summary/news-summary.jpeg
excerpt: "Using Large Language Models to generate a summary of most relevant subjects in the news today"
---

The abundance of news offered by the internet is not a new development; the amount of information we should be aware of only increases. Certainly, much if is repeated content, but how can we separate the wheat from the chaff? What we are doing in this article is to take advantage of large language models (LLMs) and [RSS](https://en.wikipedia.org/wiki/RSS) news feed to generate a short summary of what is going on in the world. For simplicity we restrict our vision of the world to a few web sites, focusing mostly North America, Europe, Middle East and Australia, but it in principle we could cover the entire world. We use the RSS feed of the New York Times, the BBC, the CNN, the Wall Street Journal, the Washington Post, Al-Jazeera, and the Australian Broadcasting Corporation. The `feedparser` package is used to parse the RSS feed.


```python
from datetime import datetime
import feedparser
```


```python
def parse_feed(label, url, links):
    parsed_entries = []
    feed = feedparser.parse(url)
    for i, entry in enumerate(feed.entries):
        if hasattr(entry, 'summary'):
            summary = entry.summary
        elif hasattr(entry, 'description'):
            summary = entry.description
        else:
            continue
        published = None
        for fmt in ['%a, %d %b %Y %H:%M:%S %z', '%a, %d %b %Y %H:%M:%S %Z']:
            try:
                published = datetime.strptime(entry.published, fmt).date()
                break
            except:
                pass
        assert published is not None
        if (datetime.now().date() - published).days > 1:
            continue
        parsed_entry = '<entry>'
        parsed_entry += f'<title>{entry.title}</title><summary>{summary}</summary><reference>{label}#{i}</reference>'
        if hasattr(entry, 'tags'):
            for tag in entry.tags:
                parsed_entry += '<tag>' + tag['term'] + "</tag>"
            parsed_entry += '</entry>'
        parsed_entries.append(parsed_entry)
        links[label][i] = entry.link
    return parsed_entries
```

Since we don't have a subscription to those sources, we limit ourselves to the provided title and summary. If the article has tags, we take them as well.


```python
from collections import defaultdict
links = defaultdict(lambda: {})
```


```python
nyt_entries = parse_feed('nyt', 'https://rss.nytimes.com/services/xml/rss/nyt/World.xml', links)
print(f"Found {len(nyt_entries)} entries for the New York Times feed.")
```

    Found 29 entries for the New York Times feed.
    


```python
bbc_entries = parse_feed('bbc', 'https://feeds.bbci.co.uk/news/world/rss.xml', links)
print(f"Found {len(bbc_entries)} entries for the BBC feed.")
```

    Found 23 entries for the BBC feed.
    


```python
cnn_entries = parse_feed('cnn', 'http://rss.cnn.com/rss/edition.rss', links)
print(f"Found {len(cnn_entries)} entries for the CNN feed.")
```

    Found 0 entries for the CNN feed.
    


```python
wsj_entries = parse_feed('wsj', 'https://feeds.a.dj.com/rss/RSSWorldNews.xml', links)
print(f"Found {len(wsj_entries)} entries for the Wall Street Journal feed.")
```

    Found 20 entries for the Wall Street Journal feed.
    


```python
wp_entries = parse_feed('wp', 'https://feeds.washingtonpost.com/rss/world', links)
print(f"Found {len(wp_entries)} entries for the Washington Post feed.")
```

    Found 25 entries for the Washington Post feed.
    


```python
aj_entries = parse_feed('aj', 'https://www.aljazeera.com/xml/rss/all.xml', links)
print(f"Found {len(aj_entries)} entries for the Al-Jazeera Post feed.")
```

    Found 25 entries for the Al-Jazeera Post feed.
    


```python
abc_entries = parse_feed('abc', 'https://www.abc.net.au/news/feed/2942460/rss.xml', links)
print(f"Found {len(aj_entries)} entries for the Australian Broadcasting Corporation Post feed.")
```

    Found 25 entries for the Australian Broadcasting Corporation Post feed.
    


```python
entries = \
    nyt_entries + \
    bbc_entries + \
    cnn_entries + \
    wsj_entries + \
    wp_entries + \
    aj_entries + \
    abc_entries
print(f"Total # entries: {len(entries)}.")
links = dict(links)
```

    Total # entries: 137.
    

For the LLM, we use [https://www.openai.com](OpenAI). The entries are shuffled to mix the different sources. The prompt suggests the LLM to take the point of view of a journalist experienced in international journalism, with the aim of providing the good (and succint) summary of what has happened during the day. Input is provided in XML format; output is requested in JSON format. 


```python
import os, openai
openai.api_key = os.getenv('OPENAI_API_KEY')
```


```python
import random
random.shuffle(entries)
```


```python
import json
prompt = """
Your are given a list of news reporting what is happening today in the world \
from different sources. The news are given in XML format; each node has \
a <title> node with a short description, a <summary> node with a longer one, a <link> node \
with the authoritative source, and possible one or more <tag> nodes. \
Some of the news are reported more than once, with a \
different title and text.

Your task is to define the most important subjects in the news. Each \
subject should be a few words or a short sentence. Ideally almost all \
news are included in one subject. The focus must be on events happening on the \
day rather than news that don't bring much more information.

All the output should be in json format, containing a list of 'subject'. \
This should be a list with the subject and nothing else.
""" + '\n\n'.join(entries)

completion = openai.chat.completions.create(
    model='gpt-4-1106-preview',
    messages=[
        {'role': 'system', 'content': "you are an journalist who is an expert in international news and is keen to provide a realistic and explicative overview of what is happening today"},
        {'role': 'user', 'content': prompt},
    ]
)

content = completion.choices[0].message.content

begin_tag = '```json'
end_tag = '```'
subjects = json.loads(content[content.find(begin_tag) + len(begin_tag):content.rfind(end_tag)])
subjects = subjects['subjects']
```


```python
print('Subjects:')
for subject in subjects:
    print(f'- {subject}')
```

    Subjects:
    - Ongoing Israel-Gaza Conflict and Airstrikes
    - Political Changes in Turkey and Local Elections
    - Iranian Generals Killed in Alleged Israeli Strike in Syria
    - World Central Kitchen Workers Killed in Gaza
    - Economic Developments and Stock Market Updates
    - Middle East Diplomacy: U.S. and Israeli Officials Meet Virtually
    - Reported Cybersecurity and Business Infrastructure Changes
    - Controversies and Developments Around Global Sports Events
    - International Law Enforcement and Justice Updates
    - Key Political Figures' Health and Legal Cases
    - Socio-Cultural Issues and Human Rights Developments
    - Environmental Challenges and Initiatives
    - Health Concerns Including Pandemics and Medical Conditions
    - Technological Advances and Innovations
    - Financial Market Regulations and Company-Specific News
    - National Leadership Changes and Inter-Governmental Relations
    - Humanitarian Crises and Aid Work Suspensions
    


```python
prompt = f"""
Your are given a list of news reporting what is happening today in the world \
from different sources. The news are given in XML format; each node has \
a <title> node with a short description, a <summary> node with a longer one, a <reference> node \
with the number of a reference, to which you don't have access, and possibly one or more <tag> nodes. \
Some of the news are reported more than once, with a \
different title and text.

Your task is to write a text containing the summary of all of what \
is happening in the world today. You should use the following subjects: \
{', '.join(subjects)}. Each subject will have its own paragraph; \
each paragraph is quite verbose, 5 to 10 lines, and it contains one or two links, \
taken from the <reference> nodes of the relevant news. \
Ignore news that are not important and focus on the ones that are relevant \
and worth knowning. Write extensively, \
verbose is better than terse. Do not add links to internet sites, only use \
the text that is provided to you. Do not make things up.

All the output should be in json format, containing a list of 'entry', \
each 'entry' having a 'title', a 'summary' and one or more 'links'. The provide \
summary should be quite verbose. 
""" + '\n\n'.join(entries)

completion = openai.chat.completions.create(
    model='gpt-4-1106-preview',
    messages=[
        {'role': 'system', 'content': "you are an journalist who is an expert in international news and is keen to provide a realistic and explicative overview of what is happening today"},
        {'role': 'user', 'content': prompt},
    ]
)

summary = completion.choices[0].message.content
```


```python
begin_tag = '```json'
end_tag = '```'
summary2 = json.loads(summary[summary.find(begin_tag) + len(begin_tag):summary.rfind(end_tag)])
```


```python
numbers = ['➀', '➁', '➂', '➃', '➄', '➅', '➆', '➇', '➈', '➉']
text = ""
items = summary2['entries'] if 'entries' in summary2 else summary2
for item in items:
    title = item['title']
    summary = item['summary']
    l = []
    for number, link in zip(numbers, item['links']):
        if 'reference#' not in link:
            continue
        link = link.replace('reference#', '')
        label, i = link.split('#')
        if '/' in label:
            label = label[label.rfind('/') + 1:]
        if label in links:
            l.append(f'[{number}]({links[label][int(i)]})')
    text += f"\n**{title}**\n\n{summary} {' '.join(l)}\n\n"
```

Finally, this is the result for the news of April 2nd, 2024. Results are quite good considering how little code we have to write.


```python
from IPython.display import display, Markdown
from datetime import datetime
today = datetime.strftime(datetime.now(), '%B %d, %Y')
display(Markdown(f"# News Summary for {today}\n\n" + text))
```


# News Summary for April 02, 2024


**Ongoing Israel-Gaza Conflict and Airstrikes**

The Israel-Gaza conflict has escalated with several reported deaths after an Israeli airstrike on Gaza's largest hospital, Al-Shifa, which is now said to be in ruins. Amidst these tragic circumstances, seven World Central Kitchen workers were killed by an Israeli strike, leading to a suspension of crucial aid work that exacerbates the territory's hunger crisis. Furthermore, there have been allegations that Israeli forces recently killed Iranian generals in a strike in Syria, intensifying the shadow war between the two nations. [➀](https://www.nytimes.com/2024/04/01/world/middleeast/gaza-al-shifa-hospital.html) [➁](https://www.washingtonpost.com/world/2024/04/01/world-central-kitchen-gaza-deaths-wck/) [➂](https://www.nytimes.com/2024/04/01/world/middleeast/iran-commanders-killed-syria-israel.html)


**Political Changes in Turkey and Local Elections**

Turkey has witnessed a political shakeup as the opposition's strength was demonstrated during local elections, taking away key areas from President Erdogan’s ruling party and hinting at Erdogan's waning influence. Istanbul's Mayor Imamoglu has been re-elected, dealing a blow to Erdogan's party. The victories mark a significant moment and offer a potential shift in Turkish politics. [➀](https://www.nytimes.com/2024/04/01/world/middleeast/turkey-election-results.html) [➁](https://www.nytimes.com/2024/03/31/world/middleeast/istanbul-mayor-race-turkey.html)


**Iranian Generals Killed in Alleged Israeli Strike in Syria**

The tension along the Syrian-Israeli axis surged as reports emerged that an Israeli missile strike in Damascus targeted and killed several Iranian commanders. This act could potentially escalate an already taut regional scenario and represents a significant moment in the shadow conflict straddling the region. [➀](https://www.bbc.co.uk/news/world-middle-east-68708923) [➁](https://www.wsj.com/world/middle-east/syria-iran-blame-israel-for-deadly-attack-in-damascus-1d28df42)


**World Central Kitchen Workers Killed in Gaza**

The tragic death of seven World Central Kitchen workers in Gaza, including a U.S.-Canadian citizen, has halted the charity work that feeds thousands in the area. This incident threatens to compound the humanitarian crisis in Gaza and puts a spotlight on the dire need for consistent aid in conflict zones. [➀](https://www.washingtonpost.com/world/2024/04/02/israel-hamas-war-news-gaza-palestine/) [➁](https://www.wsj.com/articles/world-central-kitchen-suspends-gaza-aid-operations-after-workers-killed-71ed846f)


**Economic Developments and Stock Market Updates**

The global economy presents a mixed picture as U.S. stocks show a divergent trend with the S&P 500 and Dow industrials seeing drops and the Nasdaq experiencing modest gains. Additionally, China's property giant Country Garden is going through financial turbulence which led to them suspending shares. [➀](https://www.wsj.com/finance/stocks/global-stocks-markets-dow-news-04-01-2024-c6fecc1d) [➁](https://www.bbc.co.uk/news/business-68710728)


**Middle East Diplomacy: U.S. and Israeli Officials Meet Virtually**

Virtual discussions between U.S. and Israeli officials took place against the backdrop of turmoil in Rafah. These talks show a continuous partnership and attempt to align strategies amidst the ongoing regional conflicts, including those in Gaza and Syria. [➀](https://www.washingtonpost.com) [➁](https://www.washingtonpost.com)


**Reported Cybersecurity and Business Infrastructure Changes**

Technology giant Microsoft has undergone structural changes by separating its Teams and Office products globally, with the European Commission having investigated for potential anti-competitive behaviour since 2020. This shift may influence the landscape of collaborative software and business communications. [➀](https://www.bbc.co.uk/news/business-68705709)


**Controversies and Developments Around Global Sports Events**

Sports events have not escaped controversy with recent developments. Florida voters will soon be weighing in on one of America's strictest abortion laws and Germany has tackled an unfortunate resemblance of their new kit option to Nazi symbolism. In the realm of sports, these issues go beyond the playfield reflecting broad socio-political concerns. [➀](https://www.bbc.co.uk/news/world-us-canada-68710223) [➁](https://www.bbc.co.uk/news/world-europe-68708981)


**International Law Enforcement and Justice Updates**

The international community has seen important law enforcement actions, notably with a journalist for Radio Free Europe/Radio Liberty now detained by Russia, and Hunter Biden's legal battles continuing as a judge rejects his bid to toss out a tax case. These cases signify a nexus of media, politics, and legal interpretations. [➀](https://www.wsj.com/world/russia/russian-court-extends-detention-of-radio-free-europe-reporter-06113c9f) [➁](https://www.aljazeera.com/economy/2024/4/2/us-judge-rejects-hunter-bidens-bid-to-toss-out-tax-case?traffic_source=rss)


**Key Political Figures' Health and Legal Cases**

Political figures have been in the spotlight due to their health issues or legal predicaments, such as Israel’s Prime Minister Benjamin Netanyahu undergoing surgery amidst political turmoil. Moreover, former President Trump posts significant bond in a New York fraud case to avoid asset seizures, reflecting the ongoing legal challenges he faces. [➀](https://www.nytimes.com/2024/03/31/world/middleeast/netanyahu-hernia-surgery.html) [➁](https://www.bbc.co.uk/news/world-us-canada-68709895)


**Socio-Cultural Issues and Human Rights Developments**

Socio-cultural challenges remain salient, as displayed by the Scottish Hate Crime Law stirring controversy over potential freedom of speech limitations and issues surrounding gender identity being reminded by the Transgender Day of Visibility. In such a diverse sociopolitical climate, legislations and cultural recognition are sources of both progress and polarization. [➀](https://www.nytimes.com/2024/04/01/world/europe/scotland-hate-crime-law.html) [➁](https://www.abc.net.au/news/2024-04-02/sam-wallace-joseph-easter-sunday-transgender-visibility-day/103658616)


**Environmental Challenges and Initiatives**

Environmental challenges have stood out significantly with technological innovations being sought to combat global warming through methods such as geoengineering. Initiatives have risen proposing audacious solutions like blocking solar rays and carbon dioxide sequestration, pushing us to contemplate whether we are on the cusp of a tech-assisted environmental revival or heading towards uncharted risks. [➀](https://www.nytimes.com/2024/03/31/climate/climate-change-carbon-capture-ccs.html)


**Health Concerns Including Pandemics and Medical Conditions**

Global health concerns persist as an individual in Texas tests positive for avian influenza, highlighting the threat of zoonotic diseases. On a different front, the world grapples with the ongoing impacts of pandemics, underscoring the urgency of international health vigilance and preparedness. [➀](https://www.wsj.com/health/healthcare/bird-flu-human-infection-texas-cattle-885b00be)


**Technological Advances and Innovations**

The landscape of innovation gains new contours with Xiaomi's entry into the electric vehicle market, indicative of tech firms diversifying into green energy solutions. Moreover, the technological impact on privacy is spotlighted as Google plans to eradicate a vast amount of web-browsing data, a move driven by privacy concerns and legal imperatives. [➀](https://www.wsj.com/articles/xiaomi-shares-jump-after-launch-of-its-first-electric-vehicle-db10dd60)


**Financial Market Regulations and Company-Specific News**

Financial markets are eyeing regulations and company-specific news closely. Lonza Group appoints a new CEO amidst a strategic evolution, and Sports Illustrated’s legal troubles surrounding missed payments draw attention to the complexities of media financing. These developments may presage further scrutiny and possible regulatory reform. [➀](https://www.wsj.com/articles/lonza-taps-siegfried-holdings-wolfgang-wienand-as-next-ceo-d0a27384) [➁](https://www.wsj.com/business/media/sports-illustrated-owner-sues-5-hour-energy-founder-over-48-8-million-in-missed-payments-a944cb38)


**National Leadership Changes and Inter-Governmental Relations**

Amidst a dynamic global political landscape, significant leadership changes have occurred, such as in the Democratic Republic of Congo with the appointment of their first woman Prime Minister, Judith Suminwa Tuluka. This appointment signifies a potential shift in political dynamics and inter-governmental relations within the region and beyond. [➀](https://www.aljazeera.com/news/2024/4/1/dr-congo-president-names-judith-suminwa-tuluka-as-first-woman-pm?traffic_source=rss)


**Humanitarian Crises and Aid Work Suspensions**

Humanitarian crises have hit a peak with Gaza’s Al-Shifa Hospital lying in ruins after Israeli raids, dramatically underscoring the intersection of conflict and health care. This has substantially impacted the humanitarian response capabilities in the region, with organizations like World Central Kitchen forced to suspend operations after workers' deaths. [➀](https://www.bbc.co.uk/news/world-middle-east-68705765) [➁](https://www.nytimes.com/2024/04/01/world/middleeast/world-central-kitchen-strike-gaza.html)



