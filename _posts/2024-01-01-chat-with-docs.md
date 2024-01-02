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
    display(Markdown('<div style="font-family: monospace; color:#880E4F">' + text + "</div><br>"))
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


```python
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
```


```python
from langchain.chains.query_constructor.base import AttributeInfo
metadata_field_info = [
    AttributeInfo(
        name="article",
        description="The article number, generally a number be between 1 and 89`",
        type="string",
    ),
]
```


```python
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name='gpt-4', temperature=0)
```

The first prompt we develop is stateless, that is each question (and answer) is independent of what was asked before.


```python
from langchain.prompts import PromptTemplate

document_prompt = """
{title}
{article}: {page_content}
"""
document_prompt = PromptTemplate(
    input_variables=["Title", "Article", "page_content"],
    template=document_prompt)

template = """
Use the following pieces of context (delimited by <ctx></ctx>) and history
and the chat history (delimited by <hs></hs>) to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use up to ten sentences maximum; refer to the articles that are used in the answer.

<ctx>
{context}
</ctx>

Question: {question}

Helpful Answer:"""
prompt = PromptTemplate(
    input_variables=["context", "question", "Title"],
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


<div style="font-family: monospace; color:#880E4F">No, the President cannot testify in a trial during his term of office. According to Article 67 of Title IX, throughout his term of office, the President shall not be required to testify before any French Court of law or Administrative authority.</div>



```python
question = "How is power shared between the President and the Prime Minister?"
answer = get_answer(question)
my_display(answer['result'])
```


<div style="font-family: monospace; color:#880E4F">The President and the Prime Minister share power in a number of ways. The President appoints the Prime Minister and can terminate his appointment if the Prime Minister tenders the resignation of the Government (Article 8). The President also presides over the Council of Ministers (Article 9) and can communicate with the two Houses of Parliament (Article 18). The Prime Minister, on the other hand, directs the actions of the Government, is responsible for national defence, ensures the implementation of legislation, and has the power to make regulations and appointments to civil and military posts (Article 21). The Prime Minister can also delegate certain powers to Ministers and deputize for the President in certain cases (Article 21). The Government, led by the Prime Minister, determines and conducts the policy of the Nation and is accountable to Parliament (Article 20).</div>



```python
question = "What is the role of the Government?"
answer = get_answer(question)
my_display(answer['result'])
```


<div style="font-family: monospace; color:#880E4F">The Government's role, as outlined in Article 20, is to determine and conduct the policy of the Nation. It has at its disposal the civil service and the armed forces. The Government is accountable to Parliament according to the terms and procedures set out in articles 49 and 50. The Government can also consult the Economic, Social and Environmental Council on any economic, social or environmental issue as per Article 70. Members of the Government are criminally liable for serious crimes or major offences committed in office, and they are tried by the Court of Justice of the Republic as per Article 68-1.</div><br>



```python
question = "What is the role of the Parliament?"
answer = get_answer(question)
my_display(answer['result'])
```


<div style="font-family: monospace; color:#880E4F">The Parliament in this context has several roles. According to Article 24, it is responsible for passing statutes, monitoring the actions of the Government, and assessing public policies. It is composed of the National Assembly and the Senate, both of which represent French Nationals living abroad. The Parliament also has the power to authorize a declaration of war as per Article 35. It can oppose modifications of the rules governing the passing of Acts of the European Union according to Article 88-7. Furthermore, the Parliament is assisted by the Cour des Comptes in monitoring Government action and assessing public policies as stated in Article 47-2.</div><br>



```python
from langchain.prompts import PromptTemplate

document_prompt = """
{title}
{article}: {page_content}
"""
document_prompt = PromptTemplate(
    input_variables=["title", "article", "page_content"],
    template=document_prompt)

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
prompt = PromptTemplate(
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
                prompt=prompt,
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


<div style="background-color: #F3E5F5; padding: 10px; width: 90%">The President of the Republic has several powers as outlined in the articles. According to Article 15, the President is the Commander-in-Chief of the Armed Forces and presides over the higher national defence councils and committees. Article 17 gives the President the power to grant pardons in an individual capacity. In Article 16, the President has the power to take measures required in case of serious and immediate threats to the institutions of the Republic, the independence of the Nation, the integrity of its territory or the fulfilment of its international commitments. Article 14 states that the President accredits ambassadors and envoys to foreign powers. Article 5 outlines that the President ensures due respect for the Constitution, the proper functioning of the public authorities and the continuity of the State. He is also the guarantor of national independence, territorial integrity and due respect for Treaties. Article 18 allows the President to communicate with the two Houses of Parliament by messages. Article 52 states that the President negotiates and ratifies treaties. Finally, Article 67 provides immunity to the President from liability for acts carried out in his official capacity.</div>



```python
answer = chat_bot.get_answer('What are the powers of the Prime Minister?')
my_display(answer['result'])
```


<div style="background-color: #F3E5F5; padding: 10px; width: 90%">The Prime Minister has several powers as outlined in the articles. According to Article 21, the Prime Minister directs the actions of the Government, is responsible for national defence, ensures the implementation of legislation, has power to make regulations, and makes appointments to civil and military posts. The Prime Minister may delegate certain powers to Ministers and may deputize for the President of the Republic in certain cases. Article 49 allows the Prime Minister to make the Government's programme or a general policy statement an issue of a vote of confidence before the National Assembly. The Prime Minister may also ask the Senate to approve a statement of general policy. According to Article 20, the Prime Minister, along with the Government, determines and conducts the policy of the Nation and is accountable to Parliament. Article 22 states that instruments of the Prime Minister shall be countersigned, where required, by the ministers responsible for their implementation.</div>



```python
answer = chat_bot.get_answer('What is the difference between the two?')
my_display(answer['result'])
```


<div style="background-color: #F3E5F5; padding: 10px; width: 90%">The President and the Prime Minister of the Republic both have significant roles, but their powers differ. The President has powers such as being the Commander-in-Chief of the Armed Forces, granting pardons, accrediting ambassadors, and negotiating and ratifying treaties. The President also ensures due respect for the Constitution, the proper functioning of public authorities, and the continuity of the State. On the other hand, the Prime Minister directs the actions of the Government, is responsible for national defence, ensures the implementation of legislation, and makes appointments to civil and military posts. The Prime Minister can also make the Government's programme or a general policy statement an issue of a vote of confidence before the National Assembly. The Prime Minister, along with the Government, determines and conducts the policy of the Nation and is accountable to Parliament.</div>



```python
answer = chat_bot.get_answer('What is the role of the National Assembly?')
my_display(answer['result'])
answer = chat_bot.get_answer('What is the role of the Senate?')
my_display(answer['result'])
answer = chat_bot.get_answer('What are the differences between the two?')
my_display(answer['result'])
```


<div style="background-color: #F3E5F5; padding: 10px; width: 90%">The National Assembly, as part of the Parliament, has the role of passing statutes, monitoring the action of the Government, and assessing public policies. Members of the National Assembly, not exceeding five hundred and seventy-seven, are elected by direct suffrage. The National Assembly also has the right to initiate legislation, along with the Prime Minister. Finance Bills and Social Security Financing Bills are tabled first before the National Assembly. Furthermore, the National Assembly can issue a reasoned opinion on the conformity of a draft proposal for a European Act with the principle of subsidiarity. If the National Assembly passes a resolution of no-confidence, or fails to endorse the Government programme or general policy statement, the Prime Minister shall tender the resignation of the Government.</div>



<div style="background-color: #F3E5F5; padding: 10px; width: 90%">The Senate, as part of the Parliament, has the role of passing statutes, monitoring the action of the Government, and assessing public policies. Senators, not exceeding three hundred and forty-eight, are elected by indirect suffrage. The Senate ensures the representation of the territorial communities of the Republic. It also has the right to initiate legislation. The Senate can issue a reasoned opinion on the conformity of a draft proposal for a European Act with the principle of subsidiarity. Furthermore, the President of the Senate is elected each time elections are held for partial renewal of the Senate.</div>



<div style="background-color: #F3E5F5; padding: 10px; width: 90%">The National Assembly and the Senate both have roles in passing statutes, monitoring the government, and assessing public policies. However, there are several differences between the two. Members of the National Assembly are elected by direct suffrage, while Senators are elected by indirect suffrage. The National Assembly has the first say on Finance Bills and Social Security Financing Bills, while the Senate ensures the representation of the territorial communities of the Republic. Furthermore, the President of the National Assembly is elected for the life of a Parliament, while the President of the Senate is elected each time elections are held for partial renewal of the Senate.</div>



```python
answer = chat_bot.get_answer('Are the two chambers equal in power?')
my_display(answer['result'])
```


<div style="background-color: #F3E5F5; padding: 10px; width: 90%">No, the two chambers are not equal in power. While both the National Assembly and the Senate have roles in passing statutes, monitoring the government, and assessing public policies, the National Assembly has the final say in the event of a disagreement between the two houses. If a joint committee fails to agree on a common text, or if the text is not passed, the Government may ask the National Assembly to reach a final decision. Furthermore, Finance Bills and Social Security Financing Bills are tabled first before the National Assembly.</div>



```python
answer = chat_bot.get_answer("""
What are the specific roles of the Senate compared to that of the National Assembly?
""")
my_display(answer['result'])
```


<div style="background-color: #F3E5F5; padding: 10px; width: 90%">The Senate has several specific roles compared to the National Assembly. The Senate is responsible for representing the territorial communities of the Republic, which is not a role of the National Assembly. Senators are elected by indirect suffrage, unlike members of the National Assembly who are elected by direct suffrage. Furthermore, the President of the Senate is elected each time elections are held for partial renewal of the Senate, unlike the President of the National Assembly who is elected for the life of a Parliament. Also, bills primarily dealing with the organisation of territorial communities are tabled first in the Senate.</div>



```python
chat_bot = ChatBot(llm, vectordb)
```


```python
answer = chat_bot.get_answer("""
What are the penal responsabilities of the member of the government?
""")
my_display(answer['result'])
```


<div style="background-color: #F3E5F5; padding: 10px; width: 90%">Members of the Government can be held criminally liable for acts performed in the holding of their office that are classified as serious crimes or other major offences at the time they were committed, as stated in Article 68-1. They are tried by the Court of Justice of the Republic, which is bound by the definition of serious crimes and other major offences and the determination of penalties as laid down by statute.</div>



```python
answer = chat_bot.get_answer("""
What are the penal status of the member of the Parliament?
""")
my_display(answer['result'])
```


<div style="background-color: #F3E5F5; padding: 10px; width: 90%">Members of Parliament cannot be prosecuted, investigated, arrested, detained, or tried for opinions expressed or votes cast in the performance of their official duties, according to Article 26. They cannot be arrested for a serious crime or other major offence without the authorization of the Bureau of the House of which they are a member, unless the crime was committed flagrante delicto or a conviction has become final. If a Member of Parliament is detained, subjected to custodial or semi-custodial measures, or prosecuted, these actions can be suspended for the duration of the session if the House of which they are a member requires it.</div>

