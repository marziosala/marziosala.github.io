---
layout: splash
permalink: /chat-with-docs/
title: "Chatting with Documents"
header:
  overlay_image: /assets/images/chat-with-docs/chat-with-docs-splash.jpeg
excerpt: "Using LangChain to develop a simple application for chatting with the French Constitution"
---

In this post we look at simple way to extract information from documents. This is often referred to as *chat with your docs* -- asking questions to your documents directly to obtain information that general-purpose large language models (LLMs) could not deliver. The idea is to split the documents into small chunks, compute the [embedding](https://en.wikipedia.org/wiki/Word_embedding) of each chunk. Then, we look for the chunks that are the closest (in some norm defined on the embedding space) to our question and prepare a prompt for the LLM based on the text of the chunks plus our question. The advantage is clear: we can use a generic LLM, that is one that has not been trained or refined on our internal body of knowledge. This means we can use confidential information when using the LLM locally.

We will use [LangChain](https://www.langchain.com/) to connect the pieces, [OpenAI](https://www.openai.com) for the LLM, and [Chroma](https://github.com/chroma-core/chroma) as vector database. The Python environment is created with the following packages: 

```powershell
python -m venv venv
./venv/Scripts/activate
pip install ipykernel ipywidgets nbconvert matplotlib openai \
    langchain unstructured markdown chromadb tiktoken lark
```

For the documents to chat with, we will use the [1958 French constitution](https://en.wikipedia.org/wiki/Constitution_of_France) in the official
[english](https://www.elysee.fr/en/french-presidency/constitution-of-4-october-1958) translation, as provided by the Élysée website. The content of the constitution was saved to a text file in Markdown format. The file contains the 89 articles, some of which have more than one part, and around 11,000 words, or roughly 15,000 tokens.


```python
from IPython.display import display, Markdown
from pathlib import Path
```

`my_display()` is a small helper function to print out the LLM output using a different font color and style. 


```python
from IPython.display import display, Markdown
def my_display(text):
    display(Markdown('<div style="font-family: monospace; color:#880E4F; padding: 10px">' + text + "</div>"))
```


```python
doc_path = Path('./french-constitution-en.md')
with open(doc_path, 'r') as f:
    doc = f.read()
```

As discussed above, the first step is to split the document into chunks. There are many strategies for doing so; here we use a class that is tuned for Markdown input and splits on titles and articles. This strategy works well for our case, where each article is of modest length and focused on a specific topic.


```python
from langchain.text_splitter import MarkdownHeaderTextSplitter
headers_to_split_on = [
    ("#", "title"),
    ("##", "article"),
]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
splits = markdown_splitter.split_text(doc)
print(f"Found {len(splits)} splits.")
```

    Found 109 splits.
    

The splitter enriches each chunk with the title of the section and the article number in the `title` and `article` fields of the metadata. This allows us to connect each chunk back to the parts of the original document where it is taken. For example, the chunk #10 will have the following metdata:


```python
s = splits[10]
s.metadata
```




    {'title': 'Title II - The President of the Republic', 'article': 'Article 10'}



We will use OpenAI embeddings: for each chunk, we compute the embedding and store it into a vector database. The database stores the emebeddings together with the chunks on disk, as specified by the `persist_directory` variable, so embeddings are not recomputed when the notebook kernel is restarted.


```python
from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()
```


```python
from langchain.vectorstores import Chroma
chroma_path = Path('./chroma')
if not chroma_path.exists():
    chroma_path.mkdir()
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=str(chroma_path),
    )
else:
    vectordb = Chroma(
        embedding_function=embedding,
        persist_directory=str(chroma_path),
    )
assert vectordb._collection.count() == len(splits)
```

The metadata can be used in the query itself; this is done with `SelfQueryRetriever` class.


```python
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="article",
        description="The article number, generally a number be between 1 and 89`",
        type="string",
    ),
]
```

The first prompt we develop is stateless, that is each question (and answer) is independent of what was asked before. The last of history prevents us from asking follow-up questions; we will add history in a moment.

Because we have added metadata, it is customary to use two prompts: `document_prompt` is the prompt template that is used to organize content in retrieved documents (where each document is one of the chunks defined above) while `prompt` is the actual prompt with our query. We use the document prompt to organize each document with a specific structure, reporting the title of the section, the article number and its content; the prompt itself describes what we want to obtain.


```python
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name='gpt-4', temperature=0)
from langchain.prompts import PromptTemplate

document_prompt = PromptTemplate(
    input_variables=["title", "article", "page_content"],
    template="""
{title}
{article}: {page_content}
""")

template = """
Use the following pieces of context (delimited by <ctx></ctx>) to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use up to ten sentences maximum; refer to the articles that are used in the answer.

<ctx>
{context}
</ctx>

Question: {question}

Helpful Answer:"""
prompt = PromptTemplate(
    input_variables=["context", "question", "title"],
    template=template)
```


```python
def get_answer(question):
    from langchain.chains import RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(
            search_type='mmr',
            search_kwargs=dict(k=10, n_k=5),
        ),
        return_source_documents=True,
        chain_type_kwargs={"document_prompt": document_prompt, "prompt": prompt},
    )

    result = qa_chain({"query": question})
    return result
```


```python
question = "Can the President testify in a trial during the mandate?"
answer = get_answer(question)
my_display(answer['result'])
```


<div style="font-family: monospace; color:#880E4F; padding: 10px">No, the President of the Republic cannot be required to testify in a trial during their term of office. According to Article 67 of Title IX, throughout their term, the President is not required to testify before any French Court of law or Administrative authority. They also cannot be the object of any civil proceedings, nor of any preferring of charges, prosecution or investigatory measures.</div>



```python
question = "How is power shared between the President and the Prime Minister?"
answer = get_answer(question)
my_display(answer['result'])
```


<div style="font-family: monospace; color:#880E4F; pad: 10px">The President and the Prime Minister share power in a number of ways. The President appoints the Prime Minister and can terminate his appointment if the Prime Minister tenders the resignation of the Government (Article 8). The President also presides over the Council of Ministers (Article 9) and can communicate with the two Houses of Parliament (Article 18). The Prime Minister, on the other hand, directs the actions of the Government, is responsible for national defence, ensures the implementation of legislation, and has the power to make regulations and appointments to civil and military posts (Article 21). The Prime Minister can also delegate certain powers to Ministers and deputize for the President in certain cases (Article 21). The Government, led by the Prime Minister, determines and conducts the policy of the Nation and is accountable to Parliament (Article 20).</div>



```python
question = "What is the role of the Government?"
answer = get_answer(question)
my_display(answer['result'])
```


<div style="font-family: monospace; color:#880E4F; pad: 10px">The Government determines and conducts the policy of the Nation, as stated in Article 20. It has at its disposal the civil service and the armed forces. The Government is accountable to Parliament according to the terms and procedures set out in articles 49 and 50. Members of the Government can be criminally liable for serious crimes or major offences committed while in office, as per Article 68-1. The Government can also consult the Economic, Social and Environmental Council on any economic, social or environmental issue, as mentioned in Article 70.</div>



```python
question = "What is the role of the Parliament?"
answer = get_answer(question)
my_display(answer['result'])
```


<div style="font-family: monospace; color:#880E4F">The Parliament in this context has several roles. According to Article 24, it is responsible for passing statutes, monitoring the actions of the Government, and assessing public policies. It is composed of the National Assembly and the Senate, both of which represent French Nationals living abroad. The Parliament also has the power to authorize a declaration of war as per Article 35. It can oppose modifications of the rules governing the passing of Acts of the European Union according to Article 88-7. Furthermore, the Parliament is assisted by the Cour des Comptes in monitoring Government action and assessing public policies as stated in Article 47-2.</div><br>


Let's add memory to our conversation. The underlying LLM per se has no memory, each query is stand-alone and independent of the previous ones. Memory is added by reporting the previous queries and answers, to allow the model to "see" what was discussed before and build the new answer on top of the old ones. It is easy, yet quite annoying, to write such memory system on our own; using LangChain one is a much preferred way. We need a new prompt, `prompt2`, which adds the previos history to the query.


```python
template = """
Use the following pieces of context (delimited by <ctx></ctx>) and history
and the chat history (delimited by <hs></hs>) to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use up to ten sentences maximum; refer to the articles that are used in the answer.

<ctx>
{context}
</ctx>

<hs>
{history}
</hs>

Question: {question}

Helpful Answer:"""
prompt2 = PromptTemplate(
    input_variables=["context", "history", "question", "title", "article"],
    template=template)
```


```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA

class ChatBot:
    def __init__(self, llm, vectordb):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.retriever=vectordb.as_retriever(
            search_type='mmr',
            search_kwargs=dict(k=10, n_k=5),
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectordb.as_retriever(
                search_type='mmr',
                search_kwargs=dict(k=10, n_k=5),
            ),
            return_source_documents=True,
            chain_type_kwargs=dict(
                document_prompt=document_prompt,
                prompt=prompt2,
                verbose=False,
                memory=ConversationBufferMemory(
                    memory_key="history",
                    input_key="question"),
                ),
        )

    def get_answer(self, question):
        return self.qa_chain({"query": question})
```


```python
chat_bot = ChatBot(llm, vectordb)
```


```python
answer = chat_bot.get_answer('What are the powers of the President?')
my_display(answer['result'])
```


<div style="font-family: monospace; color:#880E4F; padding: 10px">The President of the Republic has several powers as outlined in the articles. According to Article 5, the President ensures due respect for the Constitution, the proper functioning of public authorities, and the continuity of the State. He is also the guarantor of national independence, territorial integrity, and due respect for Treaties. Article 14 states that the President accredits ambassadors and envoys to foreign powers. Article 15 designates the President as the Commander-in-Chief of the Armed Forces. In Article 16, the President has the power to take measures required in case of serious and immediate threats to the institutions of the Republic, the independence of the Nation, the integrity of its territory, or the fulfilment of its international commitments. Article 17 vests the President with the power to grant pardons. Article 18 allows the President to communicate with the two Houses of Parliament. Article 19 requires that instruments of the President be countersigned by the Prime Minister and, where required, by the ministers concerned. Article 52 states that the President negotiates and ratifies treaties. Finally, Article 67 protects the President from liability for acts carried out in his official capacity.</div>



```python
answer = chat_bot.get_answer('What are the powers of the Prime Minister?')
my_display(answer['result'])
```


<div style="font-family: monospace; color:#880E4F; padding: 10px">The Prime Minister has several powers as outlined in the articles. According to Article 21, the Prime Minister directs the actions of the Government, is responsible for national defence, ensures the implementation of legislation, has power to make regulations, and makes appointments to civil and military posts. The Prime Minister may delegate certain powers to Ministers and may deputize for the President of the Republic in certain cases. Article 49 allows the Prime Minister to make the Government's programme or a general policy statement an issue of a vote of confidence before the National Assembly. The Prime Minister may also ask the Senate to approve a statement of general policy. Article 20 states that the Government, under the Prime Minister, determines and conducts the policy of the Nation. Finally, Article 22 requires that instruments of the Prime Minister be countersigned by the ministers responsible for their implementation.</div>



```python
answer = chat_bot.get_answer('What is the difference between the two?')
my_display(answer['result'])
```


<div style="font-family: monospace; color:#880E4F; padding: 10px">The President and the Prime Minister of the Republic both have significant roles, but their powers differ in several ways. The President is the head of state and has powers related to national independence, territorial integrity, respect for Treaties, and is the Commander-in-Chief of the Armed Forces. The President also has the power to grant pardons and negotiate and ratify treaties. On the other hand, the Prime Minister is the head of government and is responsible for directing the actions of the Government, including national defence and the implementation of legislation. The Prime Minister also has the power to make regulations and appointments to civil and military posts. While the President has more ceremonial and strategic roles, the Prime Minister is more involved in the day-to-day running of the government.</div>



```python
answer = chat_bot.get_answer('What is the role of the National Assembly?')
my_display(answer['result'])
answer = chat_bot.get_answer('What is the role of the Senate?')
my_display(answer['result'])
answer = chat_bot.get_answer('What are the differences between the two?')
my_display(answer['result'])
```


<div style="font-family: monospace; color:#880E4F; padding: 10px">The National Assembly, as outlined in Article 24, is part of the Parliament and is responsible for passing statutes, monitoring the action of the Government, and assessing public policies. Its members, not exceeding five hundred and seventy-seven, are elected by direct suffrage. The National Assembly also has a role in legislation initiation as per Article 39. Furthermore, according to Article 50, when the National Assembly passes a resolution of no-confidence, or fails to endorse the Government programme or general policy statement, the Prime Minister must tender the resignation of the Government.</div>



<div style="font-family: monospace; color:#880E4F; padding: 10px">The Senate, as outlined in Article 24, is part of the Parliament and shares responsibilities with the National Assembly for passing statutes, monitoring the action of the Government, and assessing public policies. Its members, not exceeding three hundred and forty-eight, are elected by indirect suffrage. The Senate also ensures the representation of the territorial communities of the Republic. Furthermore, according to Article 32, the President of the Senate is elected each time elections are held for partial renewal of the Senate.</div>



<div style="font-family: monospace; color:#880E4F; padding: 10px">The National Assembly and the Senate are both part of the French Parliament, but they have some differences. Members of the National Assembly are elected by direct suffrage, while Senators are elected by indirect suffrage. The National Assembly has the power to pass a resolution of no-confidence, which, if passed, requires the Prime Minister to tender the resignation of the Government, a power not explicitly given to the Senate. Furthermore, the Senate has a specific role in ensuring the representation of the territorial communities of the Republic. The President of the National Assembly is elected for the life of a Parliament, while the President of the Senate is elected each time elections are held for partial renewal of the Senate.</div>



```python
answer = chat_bot.get_answer('Are the two chambers equal in power?')
my_display(answer['result'])
```


<div style="font-family: monospace; color:#880E4F; padding: 10px">While both the National Assembly and the Senate are part of the French Parliament and share responsibilities for passing statutes, monitoring the action of the Government, and assessing public policies, they are not entirely equal in power. The National Assembly has the power to pass a resolution of no-confidence, which, if passed, requires the Prime Minister to tender the resignation of the Government, a power not explicitly given to the Senate. Furthermore, in the event of a disagreement between the two houses over a bill, the National Assembly has the final say, as outlined in Article 45. Therefore, the National Assembly holds slightly more power than the Senate.</div>



```python
answer = chat_bot.get_answer("""
What are the specific roles of the Senate compared to that of the National Assembly?
""")
my_display(answer['result'])
```


<div style="font-family: monospace; color:#880E4F; padding: 10px">The Senate and the National Assembly both share responsibilities for passing statutes, monitoring the action of the Government, and assessing public policies as part of the French Parliament. However, they have specific roles that differentiate them. The Senate, whose members are elected by indirect suffrage, has a specific role in ensuring the representation of the territorial communities of the Republic. On the other hand, the National Assembly, whose members are elected by direct suffrage, has the power to pass a resolution of no-confidence, which, if passed, requires the Prime Minister to tender the resignation of the Government, a power not explicitly given to the Senate. Furthermore, in the event of a disagreement between the two houses over a bill, the National Assembly has the final say.</div>


To conclude, LangChain provides a nice way to chatting with documents. It gives simple and clear interfaces to vector databases and provides the tools of chatting with memory.
