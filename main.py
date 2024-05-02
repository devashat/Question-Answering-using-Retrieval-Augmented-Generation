import numpy as np
import pandas as pd
import transformers
from langchain_community.llms import HuggingFaceHub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sentence_transformers import SentenceTransformer
import faiss
from transformers import T5Tokenizer, T5ForConditionalGeneration
from argparse import ArgumentParser
import os


os.system("pip install --quiet -U langchain-community")

'''
Global variables to reference in the code
'''
k_value=3
chunk_size=1000
chunk_overlap=200
hf_token = "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"


'''
Helper function for LangChain RAG
'''
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


'''
RAG with LangChain
'''
def langchain_rag(questions_path, passages_path, output_path):
    qas_pairs = []

    questions_df = pd.read_csv(questions_path)
    
    loader = CSVLoader(file_path=passages_path)
    passages_data = loader.load()

    embeddings = HuggingFaceEmbeddings()
    
    
    model = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        task="text-generation",
        model_kwargs={"temperature": 1, "max_length": 256},
        huggingfacehub_api_token=hf_token
    )

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(passages_data)
    vector_store = FAISS.from_documents(docs, embeddings)

    retriever = vector_store.as_retriever(search_kwargs={"k":k_value})
    prompt = ChatPromptTemplate.from_template("Given the context you have {context}, answer this {question}.")
    llm = model

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    for question in questions_df['question']:
        answer = rag_chain.invoke(question)
        docs = retriever.get_relevant_documents(question)

        qas_pairs.append({
            'question': question, 
            'answer': answer, 
            'sources': docs})

    answer_sources_df = pd.DataFrame(qas_pairs)
    answer_sources_df.to_csv(output_path)
    print("LangChain RAG: Done generating answers.")


'''
RAG without LangChain
'''
def rag_without_langchain(questions_path, passages_path, output_path):

    answers_and_docs = []

    passages_df = pd.read_csv(passages_path)
    questions_df = pd.read_csv(questions_path)

    documents = passages_df['context'].tolist()
    questions = questions_df['question'].tolist()

    encoder_model = SentenceTransformer('sentence-transformers/roberta-base-nli-stsb-mean-tokens')
    generative_model_name = "google/flan-t5-base"
    tokenizer = T5Tokenizer.from_pretrained(generative_model_name)
    generative_model = T5ForConditionalGeneration.from_pretrained(generative_model_name)
    
    document_embeddings = encoder_model.encode(documents)
    question_embeddings = encoder_model.encode(questions)

    dimension = document_embeddings.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(document_embeddings)

    for idx, question_embedding in enumerate(question_embeddings):
        
        D, I = index.search(np.array([question_embedding]), k_value)
        
        retrieved_docs = [documents[i] for i in I[0]]
        
        context = " ".join(retrieved_docs)
        
        prompt = f"Given the context: {context}, answer the question: {questions[idx]}."
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        
        outputs = generative_model.generate(input_ids, max_length=200, num_beams=5, early_stopping=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        answers_and_docs.append({
            "question": questions[idx],
            "answer": answer,
            "sources": retrieved_docs
        })

    answers_df = pd.DataFrame(answers_and_docs)
    answers_df.to_csv(output_path, index=False)
    
    print("RAG without LangChain: Done generating answers.")

'''
Simple QA with LangChain -- No RAG
'''
def no_rag_qa_langchain(questions_path, output_path):

    questions = pd.read_csv(questions_path)
    
    model = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        task="text-generation",
        model_kwargs={"temperature": 1, "max_length": 256},
        huggingfacehub_api_token=hf_token
    )

    prompt = ChatPromptTemplate.from_template("You are intelligent, you are great at general knowledge questions. Do your best to answer this {question}")
    model = model
    output_parser = StrOutputParser()
    qa_pairs = []
    
    chain = prompt | model | output_parser
    
    for question in questions['question']:
        answer = chain.invoke({"question": question})
    
        qa_pairs.append({
            'question': question, 
            'answer': answer})


    answers_df = pd.DataFrame(qa_pairs)
    answers_df.to_csv(output_path)

    print("No RAG QA with LangChain: Done generating answers.")






if __name__ == "__main__":
    parser = ArgumentParser("homework 3 CLI")

    parser.add_argument('--rag', action="store_true", help="Indicator to perform RAG")
    parser.add_argument('--langchain', action="store_true", help="Indicator to call LangChain code")

    parser.add_argument('--questions', help="path to questions file")
    parser.add_argument('--passages', help="path to passages file")

    parser.add_argument('--output', help="output path of predictions")

    args = parser.parse_args()

    if args.rag and args.langchain:
        print("Performing RAG with LangChain")
        
        langchain_rag(args.questions, args.passages, args.output)
        
    elif args.rag:
        print("Performing RAG without LangChain")
        
        rag_without_langchain(args.questions, args.passages, args.output)
        
        
    else:
        print("Performing no RAG QA with LangChain")
        
        no_rag_qa_langchain(args.questions, args.output)
        
        
        
    

