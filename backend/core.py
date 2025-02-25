import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain import  hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_pinecone import PineconeVectorStore
from sympy.utilities.codegen import Argument

from ingestion import embeddings

load_dotenv()


def run_llm(query: str, chat_history: List[Dict[str, Any]] =[]):
    embeddings = OllamaEmbeddings(
        model="mistral",
        temperature=0
    )

    docsearch =PineconeVectorStore(embedding=embeddings,index_name=os.environ["INDEX_NAME"])

    chatLLM = OllamaLLM(
        model="llama3.2",
        temperature=0
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")  #This is a prompt. This can be changed to other prompts
    stuff_document_chain = create_stuff_documents_chain(chatLLM,retrieval_qa_chat_prompt) #Argumentation

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")  #This is a prompt. This can be changed to other prompts
    history_aware_retriever = create_history_aware_retriever(
        llm=chatLLM, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    ) #Argumentation

    qa = create_retrieval_chain(
        retriever=history_aware_retriever  , combine_docs_chain=stuff_document_chain
    )

    result = qa.invoke(input={"input":query, "chat_history":chat_history })

    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_document": result["context"],
    }

    return new_result


if __name__=="__main__":
    res = run_llm(query="what is langchain Chain?")
    print(res)

