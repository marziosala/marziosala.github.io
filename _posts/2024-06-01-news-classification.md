---
layout: splash
permalink: /news-classification/
title: "Classify news into categories based on their headline"
header:
  overlay_image: /assets/images/news-classification/news-classification.jpeg
excerpt: "Using Large Language Models to classify news into categories based on their headline and a short summary"
---

What we aim to do in this article is to categorize, using a large language model, the news contained in the RSS feed of the [New York Times](https://www.nyt.com). We focus only on the information that is available in the RSS feed, so title, a short summary of the page, and potentially some tags; we ignore instead the content of the article itself.


```python
from datetime import datetime
import feedparser
```

The `parse_feed` function is essentially equivalent to the one we developed for the [news summary](https://marziosala.github.io/news-summary/) article, and is based on the `feedparser` project.


```python
def parse_feed(label, url):
    parsed_entries = []
    feed = feedparser.parse(url)
    for i, events in enumerate(feed.entries):
        if hasattr(events, 'summary'):
            summary = events.summary
        elif hasattr(events, 'description'):
            summary = events.description
        else:
            continue
        published = None
        for fmt in ['%a, %d %b %Y %H:%M:%S %z', '%a, %d %b %Y %H:%M:%S %Z']:
            try:
                published = datetime.strptime(events.published, fmt).date()
                break
            except:
                pass
        assert published is not None
        if (datetime.now().date() - published).days > 1:
            continue
        parsed_events = dict(
            title=events.title,
            summary=summary,
            reference=f'{label}#{i}',
            tags=[]
        )
        if hasattr(events, 'tags'):
            for tag in events.tags:
                parsed_events['tags'].append(tag['term'])
        parsed_entries.append(parsed_events)
        parsed_events['link'] = events.link
    return parsed_entries
```

Since we don't have a subscription to those sources, we limit ourselves to the provided title and summary. If the article has tags, we take them as well.


```python
from collections import defaultdict
```

We limit ourselves to the news feed of the [New York Times](https://www.nyt.com). On the day this was done, there were 30 events, which are the ones we will try to classify.


```python
events = parse_feed('nyt', 'https://rss.nytimes.com/services/xml/rss/nyt/World.xml')
print(f"Found {len(events)} events for the New York Times feed.")
```

    Found 30 events for the New York Times feed.
    

The large language model we use is from Mistral AI, and in particular `mistral-medium-latest`. Results are quite good also with `mistral-small-latest` but a bit less nouanced, and not much better with `mistral-large-latest`. The prompt itself is extremely important -- results aren't good at all without detailed explanations and examples.


```python
import json
import os
import time
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
```

As customary, the `MISTRAL_API_KEY` is loaded from the environment, which takes it (when using Visual Studio Code) from the `.env` file.


```python
api_key = os.environ["MISTRAL_API_KEY"]
client = MistralClient(api_key=api_key)
```

The function `events_as_str` is used to convert the news event, defined by a title, a summary and (generally) by one or more tags to a string; all strings are formatted as a Markdown list, which is what the model will work on.


```python
def events_as_str(events):
    retval = []
    for i, event in enumerate(events[:50]):
        s = f'- [{i + 1}] ' + event['title'] + '. ' + event['summary']
        if len(event['tags']) > 0:
            s += '. Tags: ' + ', '.join(event['tags'])
        retval.append(s)
    retval = '\n'.join(retval)
    return retval
```

Another component that is quite important are the categories: make them too wide and they are of little interest, make them too narrow and the model will end up with new, more appropriate categories. It was quite effective to define examples of each category, even if his increases the number of input tokens for the prompt. All this information is defined as a Python dictionary, where the values are the categories themselves and the keys the explanations.


```python
CATEGORIES = {
    "Politics": "government policies, international relations, political scandals",
    "Business and Economy": "market trends, corporate news, economic policies, financial reports",
    "Science and Health": "medical research, health policies, environmental science, space exploration",
    "Sports": "major tournaments, athlete profiles, game results, sports policies",
    "Entertainment": "movies, music, celebrity news, cultural events",
    "Lifestyle": "fashion, travel, food, wellness",
    "Education": "school policies, educational research, academic achievements",
    "Crime and Law": "criminal cases, legal decisions, law enforcement news",
    "Environment": "climate change, conservation efforts, natural disasters",
    "Social Issues": "human rights, social justice, community events, disability issues",
    "International News": "global events, international conflicts, diplomatic relations",
    "Local News": "community events, local government, regional issues",
    "Opinion and Editorials": "commentary, analysis, opinion pieces",
    "Science and Technology": "research breakthroughs, tech advancements, scientific discoveries",
    "Culture and Arts": "art exhibitions, literature, theater, cultural heritage",
    "Weather": "weather forecasts, climate reports, extreme weather events",
    "Accidents and Disasters": "natural disasters, accidents, emergency responses",
    "Human Interest": "personal stories, inspirational events, unique human experiences",
    "War and Conflicts": "armed conflicts, military operations, peace negotiations",
    "Religion": "religious events, interfaith dialogues, religious policies",
    "Unspecified": "anything that doesn't fit elsewhere"
}
```

Finally, this is the key function: it takes a list of news events, as returned by the `parse_feed` function, serialized them using `events_as_str`, and instructs the model on what to do. We request the output as JSON, both by asking for it in the prompt but also setting `response_format=dict(type='json_object'),` in the call to Mistral AI.


```python
def compute_categories(events):

    begin_time = time.time()

    categories_as_str = '\n'.join([f'- **{k}** ({v})' for k, v in CATEGORIES.items()])

    prompt = f"""You are an journalist that is a covering world affairs. \
Your task is to categorize events happening in the world, reported with the tags <<< and >>> \
for categories and regions of the world. You must use one of the \
following predefined categories: 

{categories_as_str}.

Alse choose one region of the world from the following regions:

Be impartial and objective. Don't make things up, if you don't know the use the "Unspecified" category. \
Do not provide explanations or notes. For each category report the \
relevance as a probability, expressed as a number between 0 and 100. Always use \
one or two categories from provided list. Only use the categories in the provided list, \
don't create new categories.

###

Examples of the input:

- [1] Live Updates: Thousands Protest in Israel After Recovery of 6 Dead Hostages. Demonstrators lashed out at Prime Minister Benjamin Netanyahu, accusing him of torpedoing efforts to secure a cease-fire in exchange for the hostages’ release. The country’s largest labor union said it would go on strike on Monday.
- [2] How Telegram Became a Playground for Criminals, Extremists and Terrorists. Drug dealers, scammers and white nationalists openly conduct business and spread toxic speech on the platform, according to a Times analysis of more than 3.2 million Telegram messages.. Tags: Computers and the Internet, Mobile Applications, Terrorism, Child Abuse and Neglect, Social Media, Right-Wing Extremism and Alt-Right, Drug Abuse and Traffic, Israel-Gaza War (2023- ), Rumors and Misinformation, Instant Messaging, Fringe Groups and Movements, Black Markets, Censorship, Corporate Social Responsibility, Start-ups, Telegram LLC, Durov, Pavel Valeryevich (1984- )
- [3] The Pivotal Decision That Led to a Resurgence of Polio. In 2016, the global health authorities removed a type of poliovirus from the oral vaccine. The virus caused a growing number of outbreaks and has now arrived in Gaza.. Tags: your-feed-science, Vaccination and Immunization, Disease Rates, Poliomyelitis, Epidemics, Children and Childhood, Medicine and Health, Paralysis, Viruses, Preventive Medicine, Developing Countries, World Health Organization, Gaza Strip

Examples of the output, in JSON format:

[
    {{"id": 1,
        "categories": ["Politics", "Social Issues"],
        "probs": [70, 30]
    }},
    {{
        "id": 2,
        "categories": ["International News", "Science and Technology"]: 
        "probs": [60, 40]
    }},
    {{
        "id": 3,
        "categories": ["Science and Health", "International News"],
        "probs": [90, 10] 
    }}
]

Make sure each input event has exactly one output, in the same order.

### Events to classify

<<<
{events_as_str(events)}
>>>
"""

    messages = [
        ChatMessage(role="user", content=prompt)
    ]

    chat_response = client.chat(
        model='mistral-large-latest',
        messages=messages,
        response_format=dict(type='json_object'),
    )

    response = chat_response.choices[0].message.content
    try:
        parsed_response = json.loads(response)
    except:
        parsed_response =  'FAILED'

    total_time = time.time() - begin_time
    tokens_per_second = (chat_response.usage.prompt_tokens + chat_response.usage.completion_tokens) / total_time 
    print(f"prompt tokens: {chat_response.usage.prompt_tokens}, " \
          f"completion tokens: {chat_response.usage.completion_tokens} " \
          f" ({tokens_per_second:.2f} tokens/s)")

    return response, parsed_response
```

In general the model understands quite well what to do and returns as many outputs as inputs. Sometimes it doesn't, returning an empty output or a list with one empty entry.


```python
response, categories = compute_categories(events)
assert len(events) == len(categories)
```

    prompt tokens: 4343, completion tokens: 1533  (137.13 tokens/s)
    

We can now print out the results: for each news event we report the title, the summary and the categories, together with its percentage. Sometimes it is difficult for a human to assign precise categories, but overall results make sense.


```python
for i, (event, category) in enumerate(zip(events, categories)):
    #assert i + 1 == category['id']
    print(event['title'])
    print(event['summary'])
    print('⇒ ', end='')
    for cat, prob in zip(category['categories'], category['probs']):
        warning = ' ⚠' if cat not in CATEGORIES else ''
        print(f'{cat} ({prob}%{warning})', end=' ')
    print()
    print()
```

    In Rural China, ‘Sisterhoods’ Demand Justice, and Cash
    Growing numbers of Chinese women are challenging a longstanding tradition that denies them village membership, and the lucrative payouts that go with it.
    ⇒ Social Issues (60%) Politics (40%) 
    
    Brazil’s X Ban Upended Digital Businesses Overnight
    The ban on Elon Musk’s X has dealt a blow to Brazilians whose livelihoods depended on internet followings they had amassed for years, and which disappeared overnight.
    ⇒ Business and Economy (60%) Science and Technology (40%) 
    
    3 Israelis Dead in Shooting at Allenby Crossing With Jordan, Israel’s Military Says
    The Allenby crossing, near the West Bank city of Jericho, has been the scene of violence in the past.
    ⇒ War and Conflicts (70%) International News (30%) 
    
    Edmundo González, Opposition Candidate, Flees Venezuela
    Edmundo González, who is widely considered to have won July’s disputed presidential election, was facing an arrest warrant.
    ⇒ Politics (70%) International News (30%) 
    
    Kuwait Turns to Power Cuts as Climate Change Strains Its Grid
    The Persian Gulf nation has instituted rolling blackouts to cope with surging summer electricity demand, stirring frustration among citizens.
    ⇒ Environment (70%) Social Issues (30%) 
    
    Iran Sent Ballistic Missiles to Russia, U.S. and European Officials Say
    U.S. and European countries had warned of sanctions if Iran provided weapons that could be used against Ukraine. President Biden’s lame-duck status could hamper a response.
    ⇒ International News (70%) Politics (30%) 
    
    Super Typhoon Yagi Makes Landfall in Vietnam After Pounding Southern China
    At least four people have died and thousands were evacuated after Yagi, one of the strongest storms to hit northern Vietnam, brought powerful winds and torrential rains.
    ⇒ Weather (70%) Accidents and Disasters (30%) 
    
    Tony Blair’s Advice on Leadership: Tend to Your Legacy
    In an interview, the former British prime minister discussed his new book ‘On Leadership,’ the dysfunction of U.S. politics, and deflected questions about Elon Musk’s influence.
    ⇒ Politics (60%) International News (40%) 
    
    In Papua New Guinea, Pope Francis Hears Plea for Climate Action
    Pope Francis is visiting Papua New Guinea, which has been exploited for its natural resources and is imperiled by rising sea levels.
    ⇒ Environment (70%) International News (30%) 
    
    Ukrainian Street Artist Documents War Against Russia, One Stark Mural at a Time
    Using ruins as his canvas, Gamlet Zinkivskyi has captured life in wartime Ukraine in dozens of grim, gripping and harshly beautiful paintings. “Broken, but invincible,” read one captioned work.
    ⇒ War and Conflicts (70%) Culture and Arts (30%) 
    
    Indonesia Is One of the World’s Biggest Sources of Catholic Priests
    A seminary on Flores, a Catholic-majority island in Indonesia, ordains so many priests that a lot of them go abroad to serve the faithful.
    ⇒ Religion (70%) Education (30%) 
    
    American Killed at West Bank Protest Was a Campus Organizer
    Her trip to the West Bank, where she was shot on Friday, was Ms. Eygi’s latest effort in years of activism that began nearly a decade ago when was still a teenager.
    ⇒ Politics (70%) Education (30%) 
    
    Struggling to Stem Extremism, Tajikistan Hunts Beards and Head Scarves
    After Tajiks were charged with a deadly attack in Moscow, the country has cracked down on signs of Islam. But experts say it’s not addressing the causes of terrorism.
    ⇒ Social Issues (60%) Politics (40%) 
    
    Family of American Slain in the West Bank Demands an Independent Inquiry
    With witnesses and Palestinian officials accusing Israeli soldiers of firing the fatal shots, “an Israeli investigation is not adequate,” the family said in a statement.
    ⇒ Politics (70%) International News (30%) 
    
    C.I.A. and MI6 Chiefs Discuss Ukraine’s Incursion Into Russia and Gaza War
    Appearing together publicly for the first time in the history of their agencies, the heads of the U.S. and British intelligence services discussed Ukraine’s incursion into Russia and the war in Gaza.
    ⇒ International News (70%) Politics (30%) 
    
    Ukrainian Forces Block Russian Advance on a Key Eastern Town
    Russia’s drive toward Pokrovsk has stalled along one part of the frontline, but its troops continue to advance in other parts of eastern Ukraine, and its long-range aerial attacks continue.
    ⇒ War and Conflicts (70%) International News (30%) 
    
    India’s Epidemic of Cow Vigilantism Unnerves Nation’s Muslims
    An unexpectedly narrow victory at the polls for Prime Minister Narendra Modi and his Hindu-first agenda has not cooled simmering sectarian tensions, as some had hoped.
    ⇒ Social Issues (60%) Politics (40%) 
    
    Israel Strikes Schools Turned Shelters in Jabaliya, Gaza Medics Say
    Israel said it had launched a “precise strike” against Hamas militants operating from two school compounds in northern Gaza, as the family of a slain American lashes out at Israel.
    ⇒ War and Conflicts (70%) International News (30%) 
    
    The U.S. Open Concludes
    Let us consider the grief of the lapsed sports fan.
    ⇒ Sports (70%) Entertainment (30%) 
    
    Online Credit Unions Offering High Interest Rates on Savings May Be Fakes
    Ontario’s financial regulator has issued warnings about three websites illegally claiming to be credit unions, but it is powerless to close them.
    ⇒ Business and Economy (70%) Crime and Law (30%) 
    
    The Pivotal Decision That Led to a Resurgence of Polio
    In 2016, the global health authorities removed a type of poliovirus from the oral vaccine. The virus caused a growing number of outbreaks and has now arrived in Gaza.
    ⇒ Science and Health (70%) International News (30%) 
    
    Meet the Team Climbing Trees in the Amazon to Better Understand Carbon Stores
    A small team in a remote corner of Colombia is surveying every tree in an effort to better understand how much planet-warming carbon the Amazon actually stores.
    ⇒ Environment (70%) Science and Technology (30%) 
    
    How Telegram Became a Playground for Criminals, Extremists and Terrorists
    Drug dealers, scammers and white nationalists openly conduct business and spread toxic speech on the platform, according to a Times analysis of more than 3.2 million Telegram messages.
    ⇒ Science and Technology (60%) International News (40%) 
    
    Twitter Changed Soccer. There’s a Risk X Will Do It Again.
    The world’s most popular pastime has been irrevocably shaped by its exposure to social media. That evolution can still go awry.
    ⇒ Sports (60%) Science and Technology (40%) 
    
    American Woman and Palestinian Girl Killed in Separate West Bank Incidents
    In a separate incident, a 13-year-old Palestinian girl was also fatally shot, part of the rising toll of West Bank violence during the war in Gaza.
    ⇒ War and Conflicts (70%) International News (30%) 
    
    Ukraine’s Zelensky Presses Western Allies for More Weapons
    The Ukrainian leader argued that escalating military pressure on Russia, combined with diplomacy, was the best way to motivate Moscow to seek peace.
    ⇒ War and Conflicts (70%) International News (30%) 
    
    Against This Mighty Paralympic Team, a Close Loss Can Feel Like a Win
    Other teams give themselves an A for effort after playing the Dutch women’s wheelchair basketball team, the favorite for the gold medal at the Paris Games.
    ⇒ Sports (70%) Social Issues (30%) 
    
    China Stops Foreign Adoptions, Ending a Complicated Chapter
    Beijing said the move was in line with international trends, as more countries have limited such adoptions. Many would-be adoptive families were left in limbo.
    ⇒ Social Issues (60%) Politics (40%) 
    
    Woman in France Testifies Against Husband Accused of Bringing Men to Rape Her
    Gisèle Pelicot spoke of the horror of being told by the police that they had evidence her husband had drugged her for years and brought men into their home to join him in raping her.
    ⇒ Crime and Law (70%) Social Issues (30%) 
    
    Boko Haram Kills at Least 170 Villagers in Nigeria Attack
    Boko Haram killed at least 170 villagers in northeastern Nigeria, community leaders say, in what is likely one of the deadliest attacks in recent years.
    ⇒ War and Conflicts (70%) International News (30%) 
    
    
