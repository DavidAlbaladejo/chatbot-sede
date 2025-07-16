import os
import logging
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
from typing import List

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)

class NomicEmbeddingsWithPrefix(Embeddings):
    """Wrapper para nomic-embed-text que añade prefijos automáticamente"""
    
    def __init__(self, base_embeddings):
        self.base_embeddings = base_embeddings
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Añade prefijo search_document: a todos los documentos"""
        prefixed_texts = [f"search_document: {text}" for text in texts]
        return self.base_embeddings.embed_documents(prefixed_texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Añade prefijo search_query: a las consultas"""
        prefixed_text = f"search_query: {text}"
        return self.base_embeddings.embed_query(prefixed_text)

class EmbeddingProvider(ABC):
    """Interfaz base para proveedores de embeddings."""

    @abstractmethod
    def embed_documents(self, texts):
        pass

    @abstractmethod
    def embed_query(self, text):
        pass


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = None, base_url: str = None):
        from langchain_ollama import OllamaEmbeddings
        model_name = model_name or os.environ.get("OLLAMA_EMBEDDING")
        base_url = base_url or os.environ.get("OLLAMA_HOST")
        if not model_name or not base_url:
            raise ValueError("OLLAMA_EMBEDDING y OLLAMA_HOST deben estar configurados")
        base_embeddings = OllamaEmbeddings(model=model_name, base_url=base_url)
        
        if "nomic-embed-text" in model_name:
            logger.info(f"Aplicando wrapper de prefijos para {model_name}")
            self.embeddings = NomicEmbeddingsWithPrefix(base_embeddings)
        else:
            self.embeddings = base_embeddings

        logger.info(f"Conectado a Ollama en {base_url} con modelo {model_name}")

    def embed_documents(self, texts):
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text):
        return self.embeddings.embed_query(text)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = None, api_key: str = None):
        from langchain_openai import OpenAIEmbeddings
        model_name = model_name or os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY debe estar configurado")
        self.embeddings = OpenAIEmbeddings(model=model_name, openai_api_key=api_key)
        logger.info(f"Conectado a OpenAI con modelo {model_name}")

    def embed_documents(self, texts):
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text):
        return self.embeddings.embed_query(text)


class EmbeddingProviderFactory:
    """Fábrica para crear el proveedor de embeddings según variable de entorno EMBEDDING_PROVIDER."""

    @staticmethod
    def create():
        provider = os.environ.get("EMBEDDING_PROVIDER", "ollama").lower()
        logger.info(f"Creando proveedor de embeddings: {provider}")
        if provider == "ollama":
            return OllamaEmbeddingProvider()
        elif provider == "openai":
            return OpenAIEmbeddingProvider()
        else:
            raise ValueError(f"Proveedor de embeddings no soportado: {provider}")
