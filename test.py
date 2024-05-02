#!/usr/bin/env python
# coding: utf-8

# # QA Using LangChain

# In[1]:


get_ipython().system('pip install --quiet -U langchain-community')


# In[2]:


import numpy as np
import pandas as pd
import transformers

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
#from langchain.document_loaders import CSVDLoader
#from langchain.vector_stores import FAISSVectorStore


# In[3]:


model = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        task="text-generation",
        model_kwargs={"temperature": 1, "max_length": 256},
        huggingfacehub_api_token="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"
    )


# In[4]:


questions = pd.read_csv('questions.csv')


# In[5]:


questions.head()


# In[6]:


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate



prompt = ChatPromptTemplate.from_template("Answer this {question}")
model = model
output_parser = StrOutputParser()
qa_pairs = []

chain = prompt | model | output_parser

for question in questions['question']:
    answer = chain.invoke({"question": question})

    qa_pairs.append({'question': question, 'answer': answer})


answers_df = pd.DataFrame(qa_pairs)


# In[7]:


answers_df.head()


# In[8]:


answers_df.to_csv('baseline_answers.csv')


# # Retrieval Augmentation with LangChain

# In[9]:


from langchain_community.document_loaders.csv_loader import CSVLoader


# In[10]:


loader = CSVLoader(file_path="./passages.csv")

data = loader.load()


# In[11]:


get_ipython().system('pip install --upgrade --quiet langchain sentence_transformers')
#!pip install faiss-cpu


# In[12]:


from langchain_community.embeddings import HuggingFaceEmbeddings


# In[13]:


embeddings = HuggingFaceEmbeddings()


# In[14]:


print(data[:1])


# In[15]:


# passages = []

# for document in data:
    
#     lines = document.page_content.split('\n')
#     context_line = next((line for line in lines if line.startswith('context:')), None)
    
#     if context_line:
#         _, context = context_line.split('context: ', 1)
#         passages.append(context)


# In[16]:


# embedded_passages = []

# for text in passages:
#     query_result = embeddings.embed_query(text)
#     embedded_passages.append(query_result)


# In[19]:


from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS


# In[20]:


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(data)
db = FAISS.from_documents(docs, embeddings)


# In[ ]:




