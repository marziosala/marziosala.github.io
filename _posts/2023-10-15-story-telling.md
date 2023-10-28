---
layout: splash
permalink: /story-telling/
title: "Story Telling with Generative AI"
header:
  overlay_image: /assets/images/story-telling/story-telling-splash.jpeg
excerpt: "A small Python application to generate a short story and read it aloud, with generated images on the background."
---

Released a little more than a year ago, ChatGPT has seen a phenomenal success, and with it the entire field of generative AI has exploded: given a small text, the *prompt*, one can easily generate text as well as images, audio and, soon, also video.

In this article we look at both text and image generation by putting together an AI-powered story teller: given a brief descriptions of our main characters and an overview of what the story should be, we want to automatically generate a short story and some images to make the story more compelling, then ask the computer to read it aloud. All of this is done in just a few lines of code! This notebook is inspired by an article by [Sean McManus](sean.co.uk) on the 'MagPi Raspberry' and is composed by two parts: in the first part we prepare the prompts for [OpenAI](openai.com) and [DeepAI](deepai.com) and collect
the text and the images of our story; in the second part we develop a super-simple application that reads the text and displays the images.

Let's start with the Python environment:

```{powershell}
python -m venv ./story-telling
./story-telling/Scripts/activate
pip install pyttsx3 requests openai pygame matplotlib jupyter-lab
```

The `opeanai` package is used to send the prompt to ChatGPT, `requests` to generate the images
using DeepAI, `pygame` to display the images and `pyttsx3` to read the text. The other two packages
aren't really essential and can be skipped.


```python
import datetime
import json
import matplotlib.pylab as plt
import openai
from pathlib import Path
import pyttsx3
import pygame
import requests
import time
```

Both OpenAI and DeepAI requires a paid subscription. The easiest is to save the keys in a file and read them as shown here.


```python
openai.api_key = open('./openai-key.txt', 'r').read()
deepai_key = open('./deepai-key.txt', 'r').read()
```

We will keep the stories in the `stories` folder; each story is identified by the timestamp of its generation. We do this because we often need to experiment by changing the prompt and looking at the results, so it is convenient to keep the history of what was obtained.


```python
STORIES = Path('./stories')
if not STORIES.exists():
    STORIES.mkdir()
else:
    assert STORIES.is_dir()
```


```python
story_dir = STORIES / datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
if not story_dir.exists():
    story_dir.mkdir()
```

What we want is a short story, composed by about 15 paragraphs, with two characters, a tiger and a lion. The type of the characters is important to steer the story; in the same way, the venue must be chosen appropriately to obtain more color in the pictures.


```python
# name of the main character
character1_name = 'Bob'
# type of the main character
character1_type = 'An intelligent, taciturn tiger with strength and courage'
# name of the second character
character2_name = 'John'
# type of the second character
character2_type = 'A brilliant, loquacious monkey with a vivid fantasy and fervent immagination'
# "Where does the story take place?"
venue = f"""in a quest to save the forest from the invasion of ravenous
locusts that are menacing to destroy it all"""
# story genre
genre = 'horror'
```

The prompt is the most important component: we need the story, but also the image prompts to generate the images. We stick to English because DeepAI at this point understands only English, and in my tests mixing languages for the text and the image prompts caused suboptimal results. We ask for the output to be in JSON format for an easier parsing. After some experiments, it turned out to work well to require specific fields.


```python
story_prompt = f"""
Please write me a short {genre} story. In this story, {character1_name} is a
{character1_type} and {character2_name} is a {character2_type}. The story
takes place {venue}. The story should have a title. Start each paragraph with 'Paragraph #'.
For each paragraph, write me an input prompt for an AI
image generator. Each image prompt must start in a new paragraph and have the words
'Image Prompt:' at the start. Choose only one book illustrator and put something in the
image prompts to say the images should be made in the style of that artist.
There should be around 10 paragraphs.
Format the output as a json file, with a 'title' field, an 'illustrator' field for the
name of the book illustrator, and several 'paragraph' field,
and within each 'paragraph' a 'text' field and an 'prompt' field. Do not number the paragraphs
or the image prompts.
"""
```


```python
output = openai.ChatCompletion.create(
    model='gpt-4',
    messages=[
        {'role': 'system', 'content': "you are a children's author"},
        {'role': 'user', 'content': story_prompt},
    ]
)

new_story = output.choices[0].message['content']
```


```python
data = json.loads(new_story)
assert 'title' in data
assert 'paragraphs' in data
for paragraph in data['paragraphs']:
    assert 'text' in paragraph
    assert 'prompt' in paragraph
print(f"# paragraphs: {len(data['paragraphs'])}")
```

    # paragraphs: 10
    


```python
title = data['title']
story_paragraphs = [p['text'] for p in data['paragraphs']]
image_prompts = [p['prompt'] for p in data['paragraphs']]
full_story = '\n\n'.join(story_paragraphs)
```


```python
print('\n\n'.join(story_paragraphs))
```

    Once upon a time, in the heart of the verdant forest, lived Bob, the intelligent, taciturn tiger, known for his unparalleled strength and courage. Alongside him was his trusted companion, John, a brilliant, loquacious monkey with a vivid fantasy and fervent imagination. Both shared an unparalleled bond; a bond that was often tested, yet never broken.
    
    News reached Bob and John about a grave threat looming over the forest. A swarm of ravenous locusts were heading their way, threatening to consume the entire forest and leave destruction in their path.
    
    Their hearts filled with trepidation, yet determined to protect their home, the duo set out on a perilous quest to save their beloved forest. The path was fraught with danger, yet their resilience remained steadfast.
    
    John, with his knack for creativity and fervent imagination, devised plans. Meanwhile, Bob, with his strength and courage, became the executor. They made an unbeatable team, their collective strength acting as the last defense against the impending catastrophe.
    
    They journeyed through steep valleys, crossed bustling rivers, and climbed towering peaks. Their courage pushed them to traverse the toughest terrains, as they tirelessly worked together to safeguard their forest.
    
    Along the way, they encountered various challenges, from treacherous landforms to hostile creatures. But each challenge only made them stronger, their dedication inspiring everyone they came across.
    
    Ultimately, the intrepid duo reached the locust swarm's source. A showdown ensued between the last defenders of the forest and the ravenous swarm. The forest held its breath as a trepid battle began.
    
    With valiant efforts and strategic prowess, Bob and John managed to divert the swarm away from the forest. The creatures of the forest rejoiced; their guardians had prevailed.
    
    The forest celebrated its saviors: Bob, the courageous tiger, and John, the imaginative monkey. From that day forward, stories of their bravery echoed through the wilderness, inspiring generations to come.
    
    In the heart of the forest, life went back to its peaceful rhythm. The stories of Bob and John's heroic quest became a testament to their undying spirit, their tale becoming a beacon of hope, courage, and friendship.
    

As a little twist, we ask ChatGPT to translate the text into a different language. Here we use italian but anything will work as well. In general the translation keeps the number of paragraphs; an explicit instruction is added to the prompt.


```python
language = 'Italian'

full_story = '\n\n'.join(story_paragraphs)

prompt = f"""
Translate the following paragraphs of a story from English to {language}. Use the style of a storyteller.
Each paragraph should be separated by an empty line. Do not change the number of paragraphs.

{full_story}
""" 

output = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[
        {'role': 'system', 'content': "you are the translator of children's stories"},
        {'role': 'user', 'content': prompt},
    ]
)

translated_story = output.choices[0].message['content']
translated_paragraphs = translated_story.split('\n\n')

assert len(translated_paragraphs) == len(story_paragraphs)
```


```python
for i, paragraph in enumerate(translated_paragraphs):
    with open(story_dir / f"{i}.txt", 'w') as f:
        f.write(paragraph) 
```

To generate the images, we take each of the image prompts just generated and call DeepAI. The output is a link, whose content is downloaded and saved to a file. 


```python
for number, image_prompt in enumerate(image_prompts):
    #image_prompt += f". {character1_name} is {character1_type} and "
    #image_prompt += f"{character2_name} is {character2_type}. They are {venue}."
    r = requests.post(
        'https://api.deepai.org/api/text2img',
        data=dict(
            text=image_prompt,
            negative_prompt="poorly drawn face, mutation, deformed, distorted face, extra limbs, bad anatomy",
            width=1080,
            height=720,
            grid_size=1,
        ),
        headers={'api-key': deepai_key},
    )
    image_url = r.json()["output_url"]
    filename = story_dir / f"{number}.jpeg"
    img_data = requests.get(image_url).content
    with open(filename, 'wb') as f:
        f.write(img_data)
```


```python
with open(story_dir / 'settings.txt', 'w') as f:
    f.write(f'{len(translated_paragraphs)}\n{len(image_prompts)}')

with open(story_dir / 'settings.txt', 'r') as f:
    num_paragraphs = int(f.readline())
    num_prompts = int(f.readline())
```


```python
paragraphs, images = [], []
for i in range(max(num_paragraphs, num_prompts)):
    paragraph_filename = story_dir / f"{i}.txt"
    if paragraph_filename.exists():
        paragraph = paragraph_filename.read_text()
        paragraphs.append(paragraph)
    image_filename = story_dir / f"{i}.jpeg"
    if image_filename.exists():
        image = pygame.image.load(image_filename)
        images.append(image)
```

We have generated all that we need; the images are reported here. Results aren't always exceptional, with too many tigers and only few monkeys. The style for paragraph #3 is wrong and out of sync with the others; #7 and #8 are too similar.


```python
i = 0
fig, axes = plt.subplots(figsize=(20, 7), nrows=2, ncols=5)
axes = axes.flatten()
for i, ax in enumerate(axes):
    ax.imshow(plt.imread(story_dir / f"{i}.jpeg"))
    ax.text(10, 100, f"#{i}", color='cyan', fontdict=dict(size=20))
    ax.axis('off')
```


    
![png](/assets/images/story-telling/story-telling-1.png)
    


The final step is a small application that displays the images and reads the text aloud. On Windows, the corresponding language pack must be installed or results will be wierd. The `pygame` application itself is basic and only used to show the images.


```python
engine = pyttsx3.init()
engine.setProperty('voice', language.lower())
```


```python
window_width, window_height = 1080, 720
pygame.init()
surface = pygame.display.set_mode((window_width, window_height))
pygame.mouse.set_visible(False)
voice = pyttsx3.init()
voice.setProperty('rate', 170)

for paragraph, image in zip(paragraphs, images):
    image = pygame.transform.scale(image, (window_width, window_height))
    surface.blit(image, (0, 0))
    pygame.display.update()
    voice.say(paragraph)
    voice.runAndWait()
    time.sleep(1)
time.sleep(1)
voice.say('Questa è la fine della storia. Spero che vi sia piaciuta!')
voice.runAndWait()
time.sleep(2)
pygame.quit()
```

Not quite a masterpiece, yet a surprising result for just a few lines for code!
