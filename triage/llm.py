from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())  

def get_llm(model_name: str = "gpt-5-nano", temperature: float = 0.0):
    """
    Get a language model instance based on the specified model name and temperature.
    """
    model = ChatOpenAI(model_name=model_name, temperature=temperature)
    return model

def get_embeddings(model_name: str = "text-embedding-3-small"):
    """
    Get an embeddings model instance based on the specified model name.
    """
    embeddings = OpenAIEmbeddings(model=model_name)
    return embeddings