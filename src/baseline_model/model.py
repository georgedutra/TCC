import sys
sys.path.append("..")  # Adiciona o diretório pai ao sys.path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from utils.timeit import timeit

INDEX_SAVE_PATH = "/disco/indexes"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cache global de modelos e vectorstore
_model_cache = {}
_vectorstore_cache = None

def clear_cache():
    """Clear all cached models and vectorstore to free memory."""
    global _model_cache, _vectorstore_cache
    _model_cache.clear()
    _vectorstore_cache = None

def get_vectorstore():
    """Get FAISS index vectorstore for similarity search (cached).

    Returns:
        FAISS: The FAISS vectorstore.
    """
    global _vectorstore_cache
    
    if _vectorstore_cache is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _vectorstore_cache = FAISS.load_local(
            INDEX_SAVE_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    
    return _vectorstore_cache

def get_llm(model_name: str = "mistral:7b",
            overwrite: bool = False,
            **kwargs):
    """Get or create cached LLM instance.
    
    Args:
        model_name (str): The name of the model to use.
        overwrite (bool): Whether to overwrite the cached model.
        **kwargs: Additional keyword arguments for Ollama configuration.
    
    Returns:
        Ollama: The cached LLM instance.
    """
    cache_key = f"{model_name}"

    if cache_key not in _model_cache or overwrite:
        default_conf = {
            "temperature": 0.1, 
            "num_ctx": 7000,
            "num_thread": None, # Use all available threads
            "num_gpu": 0, # Force CPU usage
            "top_k": 40,
            "top_p": 0.9
        }
        default_conf.update(kwargs)
        _model_cache[cache_key] = Ollama(
            model=model_name,
            **default_conf
        )
    
    return _model_cache[cache_key]

@timeit
def respond(prompt: str, 
            model_name: str = "mistral:7b", 
            temperature: float = 0.1) -> str:
    """Generate a response from the LLM (using cached model).

    Args:
        prompt (str): The input prompt.
        model_name (str, optional): The name of the model to use. Defaults to "mistral:7b".
        temperature (float, optional): The temperature to use for sampling. Defaults to 0.1.

    Returns:
        str: The generated response.
    """
    llm = get_llm(model_name, temperature)
    response = llm.invoke(prompt)
    return response

@timeit
def rag_respond(question: str,
                model_name: str = "phi3:mini",
                k=3):
    """Generate RAG response using cached vectorstore and model.
    
    Args:
        question (str): The input question.
        model_name (str): The name of the model to use.
        k (int): The number of documents to retrieve from the vectorstore.

    Returns:
        tuple: (response, sources)
    """
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
            """Responda em português brasileiro de forma concisa.
            Use o contexto fornecido. Se não souber, diga que não sabe. 
            Contexto: {context}"""),
        ("human", "{input}")
    ])

    # Context is chunk_size * amount of chunks used
    llm = get_llm(model_name, temperature=0.1, num_ctx=7000 * k) 
    chain = prompt_template | llm

    response = chain.invoke({
        "context": context,
        "input": question
    })

    return response
