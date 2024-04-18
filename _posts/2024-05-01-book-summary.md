---
layout: splash
permalink: /book-summary/
title: "Writing a Short Essay to Summarize and Explain a Book"
header:
  overlay_image: /assets/images/book-summary/book-summary-splash.png
excerpt: "Write a short essay on a masterpiece of modern literature, Frank Kafka's book 'The Process'"
---

```python
import lxml.etree as ET
```


```python
def cleanup(chapter):
    paragraphs = []
    text = ""
    for line in chapter.split('\n'):
        if len(line) == 0 and len(text) > 0:
            paragraphs.append(text.strip())
            text = ""
        else:
            text += line + ' '
    if len(text) > 0:
        paragraphs.append(text.strip())
    return '\n\n'.join(paragraphs)
```


```python
book = ET.parse("./book.xml")
titles = chapters = [chapter.attrib['title'].strip() for chapter in book.findall('chapter')]
subtitles = chapters = [chapter.attrib['subtitle'].strip() for chapter in book.findall('chapter')]
chapters = [cleanup(chapter.text) for chapter in book.findall('chapter')]
print(f"Found {len(chapters)} chapters.")
```

    Found 10 chapters.
    


```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
total = 0
for title, chapter in zip(titles, chapters):
    word_counts = vectorizer.fit_transform([chapter])
    total += word_counts.sum()
    print(f"{title:15s}⭢ {word_counts.sum():,d} words")
print(f"{'TOTAL':15s}⭢ {total:,d} words")
```

    Chapter One    ⭢ 10,729 words
    Chapter Two    ⭢ 6,189 words
    Chapter Three  ⭢ 9,105 words
    Chapter Four   ⭢ 2,839 words
    Chapter Five   ⭢ 2,625 words
    Chapter Six    ⭢ 7,869 words
    Chapter Seven  ⭢ 19,182 words
    Chapter Eight  ⭢ 11,376 words
    Chapter Nine   ⭢ 8,853 words
    Chapter Ten    ⭢ 2,010 words
    TOTAL          ⭢ 80,777 words
    


```python
import pandas as pd
import textstat
```


```python
rows = []
for chapter in chapters:
    rows.append({
        'Flesch Reading Ease': textstat.flesch_reading_ease(chapter),
        'Flesch-Kincaid': textstat.flesch_kincaid_grade(chapter),
        'Fog Scale': textstat.gunning_fog(chapter),
        'SMOG Index': textstat.smog_index(chapter),
        'Automated Readability Index': textstat.automated_readability_index(chapter),
        'Coleman-Liau Index': textstat.coleman_liau_index(chapter),
    })
metrics = pd.DataFrame.from_dict(rows)
metrics.index = titles
metrics
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
      <th>Flesch Reading Ease</th>
      <th>Flesch-Kincaid</th>
      <th>Fog Scale</th>
      <th>SMOG Index</th>
      <th>Automated Readability Index</th>
      <th>Coleman-Liau Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Chapter One</th>
      <td>79.50</td>
      <td>6.4</td>
      <td>7.88</td>
      <td>8.8</td>
      <td>7.6</td>
      <td>6.32</td>
    </tr>
    <tr>
      <th>Chapter Two</th>
      <td>75.03</td>
      <td>8.1</td>
      <td>10.05</td>
      <td>9.9</td>
      <td>9.9</td>
      <td>7.20</td>
    </tr>
    <tr>
      <th>Chapter Three</th>
      <td>77.16</td>
      <td>7.3</td>
      <td>8.82</td>
      <td>9.1</td>
      <td>8.4</td>
      <td>6.21</td>
    </tr>
    <tr>
      <th>Chapter Four</th>
      <td>81.22</td>
      <td>5.8</td>
      <td>7.57</td>
      <td>8.2</td>
      <td>6.8</td>
      <td>6.55</td>
    </tr>
    <tr>
      <th>Chapter Five</th>
      <td>76.15</td>
      <td>7.7</td>
      <td>9.47</td>
      <td>9.2</td>
      <td>9.2</td>
      <td>6.67</td>
    </tr>
    <tr>
      <th>Chapter Six</th>
      <td>79.19</td>
      <td>6.5</td>
      <td>8.04</td>
      <td>8.5</td>
      <td>7.7</td>
      <td>6.15</td>
    </tr>
    <tr>
      <th>Chapter Seven</th>
      <td>76.45</td>
      <td>7.6</td>
      <td>9.10</td>
      <td>10.1</td>
      <td>9.6</td>
      <td>7.54</td>
    </tr>
    <tr>
      <th>Chapter Eight</th>
      <td>80.72</td>
      <td>6.0</td>
      <td>7.43</td>
      <td>8.7</td>
      <td>7.2</td>
      <td>6.44</td>
    </tr>
    <tr>
      <th>Chapter Nine</th>
      <td>76.86</td>
      <td>7.4</td>
      <td>9.07</td>
      <td>9.6</td>
      <td>8.8</td>
      <td>6.79</td>
    </tr>
    <tr>
      <th>Chapter Ten</th>
      <td>80.01</td>
      <td>6.2</td>
      <td>7.92</td>
      <td>8.8</td>
      <td>7.3</td>
      <td>6.67</td>
    </tr>
  </tbody>
</table>
</div>




```python
import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
api_key = os.environ["MISTRAL_API_KEY"]
client = MistralClient(api_key=api_key)
```


```python
model = 'mistral-large-latest'
```


```python
PROMPT = """
You are a professional writer; your main area of expertise is modern literature. Your task \
is to analyze the text below, which is one chapter of Franz Kafka's novel "The Trial." \
Describe the characters, the location where the story unrolls, \
and highlight the atmosphere and the feeling that the author wants to convey. \
Since this is only one chapter of the book, don't provide conclusions, they will be \
added later. Don't use any prior knowledge of this text; you analysis should be based only on the text. \
The output should be plain text.
""".strip()
```


```python
def summarize(chapter, prev_summary=None):
    content = f"""
{PROMPT}

<<<
{chapter}
>>>
    """

    chat_response = client.chat(
        model=model,
        messages=[ChatMessage(role="user", content=content.strip())],
    )

    iter_0 = chat_response.choices[0].message.content

    # second iteration

    content = f"""
{PROMPT}

This is the chapter:
<<<
{chapter}
>>>

This is a first draft; take it and improve it as needed.
<<<
{iter_0}
>>>
    """

    if prev_summary is not None:
        content += f"""

This is the summary of the previous chapter, you should not repeat what was already said:
<<<
{prev_summary}
>>>
"""
        
    chat_response = client.chat(
        model=model,
        messages=[ChatMessage(role="user", content=content.strip())],
    )

    iter_1 = chat_response.choices[0].message.content

    return iter_0, iter_1
```


```python
old_summary = iter_1
```


```python
iter_0, iter_1 = summarize(chapters[0])
```


```python
from IPython.display import display, Markdown
display(Markdown(iter_0))
```


The chapter is set in Josef K.'s apartment, specifically in his bedroom and the adjoining living room belonging to his landlady, Mrs. Grubach. The atmosphere is one of confusion, disorientation, and invasion of privacy as K. is arrested without explanation by two unidentified men. The tone is also somewhat absurd, with the men acting in a manner that suggests they are both in charge and yet subservient to some unseen authority. The characters include Josef K., the protagonist, who is a bank employee arrested for an unspecified crime; Mrs. Grubach, his landlady; and the two unnamed men who arrest him. The author's intention seems to be to convey a sense of the absurdity and arbitrariness of the legal system and the powerlessness of the individual in the face of such a system.



```python
display(Markdown(iter_1))
```


The chapter is set in Josef K.'s room and the adjacent living room of his landlady, Mrs. Grubach. The atmosphere is one of confusion, disbelief, and invasion of personal space, as K. is arrested without explanation by two strange men. The tone is also somewhat surreal, with the men acting in a manner that is both authoritative and subservient to some unseen power. The characters include Josef K., the protagonist, who is a bank employee arrested for an unknown reason; Mrs. Grubach, his landlady; and the two unnamed men who arrest him. The author's intention seems to be to convey a sense of the absurdity and incomprehensibility of the legal system and the powerlessness of the individual when confronted with such a system. The chapter raises questions about the nature of guilt and the arbitrariness of authority, as K. is arrested without any explanation, and the men who arrest him seem to have no clear understanding of the charges against him. The text also explores the theme of the individual's struggle to assert their own agency and autonomy in the face of an oppressive and inscrutable system.


We iterate over the remaining chapters; the summary of the previous chapter is given in input, prompting the model not to repeat what was said before. We don't pass all the previous chapters to reduce the number of used tokens.


```python
essay = [iter_1]
for chapter in chapters[1:]:
    iter_0, iter_1 = summarize(chapter, prev_summary=iter_1)
    essay.append(iter_1)
```


```python
import pickle
from pathlib import Path
filename = Path('./essay.pickle')
if not filename.exists():
    with open(filename, 'wb') as f:
        pickle.dump(essay, f)
else:
    with open(filename, 'rb') as f:
        essay = pickle.load(f)
```


```python
content = """
You are a professional writer who has to write a short essay on Frank Kafka's book \
"The Process", which has been summerized and it reported below. \
Your goal is to add a final chapter to explain the importance of this writing, \
put it in context with the other books written by the author. Do not \
repeat the content of the summary, rather build on top of it to offer more depth \
in the analysis. The summary is provided in Markdown format. The output should \
be plain text without any header or section title.

<<<
"""
for title, subtitle, summary in zip(titles, subtitles, essay):
    content += f"## {title}\n### {subtitle}\n\n{summary}\n\n"
content += ">>>"
```

Let's check the number of tokens for our final query.


```python
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
tokenizer = MistralTokenizer.from_model(model)
tokenized = tokenizer.encode_chat_completion(
    ChatCompletionRequest(
        messages=[
            UserMessage(content=content),
        ],
        model=model
    )
)

print(f"# of tokens: {len(tokenized.tokens)}")
```

    # of tokens: 4037
    


```python
chat_response = client.chat(
    model=model,
    messages=[ChatMessage(role="user", content=content)],
)

conclusions = chat_response.choices[0].message.content
```


```python
conclusions = final_chapter
```


```python
from markdown_pdf import MarkdownPdf, Section
pdf = MarkdownPdf(toc_level=2)
content = '# The Process\n\n'
for title, subtitle, mini_chapter in zip(titles, subtitles, essay):
    content += f"## {title}: {subtitle}\n\n{mini_chapter}\n\n"
content += f"## Conclusions\n{conclusions}"
pdf.add_section(Section(content))
pdf.meta["title"] = "The Process, LLM Version"
pdf.meta["author"] = "Frank Kafka"
pdf.save("essay.pdf")
```
