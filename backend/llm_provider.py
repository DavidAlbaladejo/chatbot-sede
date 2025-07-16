import os
import logging
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)

def dict_to_message(d):
    role = d.get("role", "user")
    content = d.get("content", "")
    if role == "system":
        return SystemMessage(content=content)
    elif role == "assistant":
        return AIMessage(content=content)
    else:
        return HumanMessage(content=content)

class LLMProvider(ABC):
    @abstractmethod
    def chat(self, messages):
        """Recibe lista de mensajes (dicts) y devuelve respuesta o generador para streaming."""
        pass

class OllamaLLMProvider(LLMProvider):
    def __init__(self, model_name: str = None, base_url: str = None, temperature: float = 0.5):
        from langchain_ollama import ChatOllama
        model_name = model_name or os.environ.get("OLLAMA_MODEL")
        base_url = base_url or os.environ.get("OLLAMA_HOST")
        if not model_name or not base_url:
            raise ValueError("OLLAMA_MODEL y OLLAMA_HOST deben estar configurados")
        self.llm = ChatOllama(model=model_name, base_url=base_url, disable_streaming=False, num_ctx=32768, temperature=temperature)
        logger.info(f"Conectado a Ollama en {base_url} con modelo {model_name} y temperatura {temperature}")

    def chat(self, messages, config=None):
        messages_objs = [dict_to_message(m) for m in messages]
        # Retorna un generador que produce fragmentos de texto conforme se generan
        stream = self.llm.stream(messages_objs, config)

        def generator():
            full_response = ""
            for chunk in stream:
                full_response += chunk.content
                yield chunk.content  # # Cambio a yield chunk.content para token a token
        return generator()

class OpenAILLMProvider(LLMProvider):
    def __init__(self, model_name: str = None, api_key: str = None, temperature: float = 0.5):
        from langchain_openai import ChatOpenAI
        model_name = model_name or os.environ.get("OPENAI_MODEL", "gpt-4")
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY debe estar configurado")
        self.llm = ChatOpenAI(model_name=model_name, openai_api_key=api_key, streaming=True, temperature=temperature, timeout=300)
        logger.info(f"Conectado a OpenAI con modelo {model_name} y temperatura {temperature}")

    def chat(self, messages, config=None):
        messages_objs = [dict_to_message(m) for m in messages]
        # Retorna un generador que produce fragmentos de texto conforme se generan
        stream = self.llm.stream(messages_objs, config)

        def generator():
            full_response = ""
            for chunk in stream:
                full_response += chunk.content
                yield chunk.content  # Cambio a yield chunk.content para token a token
        return generator()

class AzureLLMProvider(LLMProvider):
    def __init__(self, model_name: str = None, api_key: str = None, temperature: float = 0.5):
        from langchain_openai import AzureChatOpenAI
        model_name = model_name or os.environ.get("AZURE_MODEL", "gpt-4o-mini")
        azure_endpoint = os.environ.get("AZURE_ENDPOINT", "https://your-azure-endpoint.openai.azure.com/")
        api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY debe estar configurado")
        self.llm = AzureChatOpenAI(model_name=model_name, azure_endpoint=azure_endpoint, 
                                   api_key=api_key, api_version=api_version, stream_usage=True, temperature=temperature, timeout=300
                                   )
        logger.info(f"Conectado a Azure con modelo {model_name} y temperatura {temperature}")

    def chat(self, messages, config=None):
        messages_objs = [dict_to_message(m) for m in messages]
        # Retorna un generador que produce fragmentos de texto conforme se generan
        stream = self.llm.stream(messages_objs, config)

        def generator():
            full_response = ""
            for chunk in stream:
                full_response += chunk.content
                yield chunk.content  # Cambio a yield chunk.content para token a token
        return generator()

class LLMProviderFactory:
    @staticmethod
    def create(temperature: float = 0.5):
        provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
        logger.info(f"Creando proveedor LLM: {provider}")
        if provider == "ollama":
            return OllamaLLMProvider(temperature=temperature)
        elif provider == "openai":
            return OpenAILLMProvider(temperature=temperature)
        elif provider == "azure":
            return AzureLLMProvider(temperature=temperature)
        else:
            raise ValueError(f"Proveedor LLM no soportado: {provider}")
