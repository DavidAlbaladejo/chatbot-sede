import asyncio
import os
import logging
from dotenv import load_dotenv
from langchain_core.runnables.config import RunnableConfig
from langgraph.func import task, entrypoint
from langgraph.checkpoint.memory import MemorySaver
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langsmith import Client
from .embedding_provider import EmbeddingProviderFactory
from .llm_provider import LLMProviderFactory

load_dotenv()

# Configura nivel de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)

PERSIST_DIRECTORY = os.environ.get("PERSIST_DIRECTORY", "./.chroma")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "rag-sede")
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")

embedding_provider = EmbeddingProviderFactory.create()
embeddings = embedding_provider.embeddings

llm_provider = LLMProviderFactory.create()
llm_chat = llm_provider.chat  # Método que devuelve generador para streaming
llm = llm_provider.llm  # LLM completo para uso en prompts

checkpointer = MemorySaver()

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
)

# Prompt principal del sistema, con historial conversacional
response_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        # INSTRUCCIONES DE IDIOMA (MÁXIMA PRIORIDAD)
        - DETECTA el idioma de la pregunta del ciudadano
        - RESPONDE SIEMPRE en el MISMO idioma de la pregunta
        - Si la pregunta es en inglés → responde en inglés
        - Si la pregunta es en español → responde en español
        
        # SOBRE TI
        - Eres un asistente especializado en trámites del Ayuntamiento de Murcia
        - Tu nombre es Rosi
        
        # CÓMO RESPONDER
        - Usa el contexto proporcionado y el historial de la conversación
        - Responde de forma clara y concisa, en lenguaje natural
        - Puede haber varios procedimientos en el contexto, usa los más relevantes para responder la pregunta del ciudadano
        - Si no sabes la respuesta, dilo claramente
        - No inventes información
        - Incluye siempre la URL de los procedimientos a los que haces referencia en tu respuesta (Source)
        - Si la pregunta es una queja o sugerencia, redirige a: https://sede.murcia.es/ficha-procedimiento/12
        - No respondas a preguntas no relacionadas con trámites de la Sede Electrónica
    """),
    MessagesPlaceholder(variable_name="chat_history", n_messages=10),
    ("human", "Pregunta del ciudadano: {question}\nContexto: {context}"),
])

# Reformulación del prompt para mejorar la claridad de las preguntas, sobre todo para la búsqueda de documentos
reformulation_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history", n_messages=10),  # últimos 5 mensajes (IA - Usuario x 5)
    ("system", """
        Eres un asistente especializado en trámites del Ayuntamiento de Murcia.
        Reformula la siguiente pregunta para que sea clara y autónoma, incorporando el contexto relevante del historial de mensajes.
        Usa el mismo idioma que la pregunta original.
        Intenta que sea lo más relevante posible para la búsqueda de documentos relacionados con la pregunta.
        Si la pregunta ya es completa, no la modifiques.
    """),
    ("human", "{question}")
])


# Prompt para evaluar la respuesta generada
eval_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history", n_messages=2),  # últimos 5 mensajes (IA - Usuario x 3)
    ("system", """
        Eres un asistente especializado en evaluar las respuestas a las preguntas de los ciudadanos.
        Si el ciudadano pregunta cosas totalmente irrelevantes a trámites de la Sede Electrónica del Ayuntamiento de Murcia, lo correcto es que la respuesta sea ofrecerle ayuda sobre trámites.
        Usa siempre el mismo idioma que la pregunta original del ciudadano, independientemente del idioma del contexto o de la respuesta del asistente.
        ¿Es correcta y relevante la respuesta a la pregunta del ciudadano? 
        Responde con 'Sí' o 'No' y justifica brevemente.
    """),
    ("human", "Pregunta del ciudadano:{question}\nRespuesta: {answer}\n")
])

# Recuperación de documentos relevantes basados en la pregunta del ciudadano
async def retrieve_docs(question: str, chat_history: list, reformulate: bool = False, execution_config: RunnableConfig = None):
    try:
        global vectorstore  # Asegura que usa la variable global
        
        # Reformular la pregunta si es necesario
        if reformulate:
            embed_instruction = await reformulate_query(question, chat_history, execution_config)
        else:
            embed_instruction = question
        results = await vectorstore.asimilarity_search_with_relevance_scores(
                embed_instruction, k=10, 
        )
        logger.info(f"async retrieve_docs: Documentos recuperados: {len(results) if results else 0}")
        for doc, score in results:
            logger.info(f"Documento recuperado: {doc.metadata.get('Procedimiento')} {doc.metadata.get('source', 'URL no disponible')}, Score: {score}")
        return results or []
    except Exception as e:
        logger.error(f"Error en recuperación: {e}")
        return []

# Reformula la pregunta para mejorar la claridad y relevancia
async def reformulate_query(question: str, chat_history: list, execution_config: RunnableConfig = None):
    try:
        # Obtener mensajes formateados con historial incluido
        prompt_text = reformulation_prompt.format(
            question=question,
            chat_history=chat_history
        )
        # Llamar al LLM con los mensajes formateados
        messages = [{"role": "user", "content": prompt_text}]

        response_chunks = llm_chat(messages, execution_config)

        # Acumular TODOS los chunks, no solo el último
        full_response = ""
        async for chunk in async_generator_wrapper(response_chunks):
            # Acumular el contenido de cada chunk
            if hasattr(chunk, 'content'):
                full_response += chunk.content
            else:
                full_response += chunk
        
        # Añadir logging para verificar la respuesta completa
        logger.info(f"Pregunta reformulada completa: {full_response.strip()}")
        return full_response.strip()
    except Exception as e:
        logger.error(f"Error en reformulación de pregunta: {e}")
        return question

# Genera una respuesta a la pregunta del ciudadano usando el contexto recuperado
async def generate_answer(question: str, context: str, chat_history: list = None, execution_config: RunnableConfig = None):
    try:
        
        prompt_text = response_prompt.format(
            question=question,
            context=context,
            chat_history=chat_history
        )
        
        messages = [{"role": "user", "content": prompt_text}]

        response_chunks = llm_chat(messages,execution_config)

        async for chunk in async_generator_wrapper(response_chunks):
            yield chunk  # Permite streaming incremental

    except Exception as e:
        logger.error(f"Error en generación: {e}")
        yield "Error al generar respuesta."

# Evaluación de la respuesta generada comparándola con la pregunta original
async def evaluate_answer(question: str, answer: str, chat_history: list = None, execution_config: RunnableConfig = None):
    prompt_text = eval_prompt.format(question=question, answer=answer, chat_history=chat_history)
    messages = [{"role": "user", "content": prompt_text}]
    try:
        response_chunks = llm_chat(messages, execution_config)
        full_response = ""
        async for chunk in async_generator_wrapper(response_chunks):
            full_response += chunk.content if hasattr(chunk, 'content') else chunk

        return full_response
    except Exception as e:
        logger.error(f"Error en evaluación: {e}")
        return "Error al evaluar respuesta."

# Asynchronous generator wrapper para manejar el streaming de respuestas
async def async_generator_wrapper(gen):
    for item in gen:
        yield item

@task
async def retrieval_task(state,**kwargs):
    question = state.get("question")
    if not question:
        raise ValueError("No se encontró 'question' en el estado")
    chat_history = state.get("chat_history")
    config = kwargs.get('config', None)

    # No reformulamos si no hay al menos tres mensajes, el primero incial del asistente y el siguiente del usuario y la primera respuesta del asistente.
    if not chat_history or len(chat_history) < 3: 
        results = await retrieve_docs(question, chat_history, False, config)
    else:
        results = await retrieve_docs(question, chat_history, True, config)

    logger.info(f"Documentos recuperados: {len(results) if results else 0}")
    context_text = ""
    sources_text = ""
    scores_text = ""
    if results:
        for doc, score in results:
            if score > 0.1:
                context_text = context_text + "\n Procedimiento:" + doc.metadata.get("Procedimiento", "Nombre desconocido")
                context_text = context_text + "\n Departamento:" + doc.metadata.get("Departamento", "Departamento desconocido")
                context_text = context_text + "\n ID:" + str(doc.metadata.get("ID", "ID desconocido"))
                context_text = context_text + "\n Source:" + doc.metadata.get("source", "URL no disponible")
                context_text = context_text + "\n Cotenido:" + doc.page_content
                context_text = context_text + "\n FIN de procedimiento"
                sources = doc.metadata.get("source", "URL no disponible")
                sources_text = sources_text + "\n" + sources
                scores_text = scores_text + "\n" + str(score)
    state["context"] = context_text
    state["sources"] = sources_text
    state["scores_threshold"] = scores_text

    return state

@task
async def generation_task(state, **kwargs):

    if state is None:
        raise ValueError("Generation task recibió estado None")
    question = state.get("question")
    if not question:
        raise ValueError("No se encontró 'question' en el estado en generation_task")
    context = state.get("context", "")

    chat_history = state.get("chat_history", [])

    config = kwargs.get('config', None)

    answer = ""
    async for chunk in generate_answer(question=question, context=context, chat_history=chat_history, execution_config=config):
        answer += chunk.content if hasattr(chunk, 'content') else chunk
    state["answer"] = answer

    return state

@task
async def evaluation_task(state, **kwargs):

    if state is None or "question" not in state:
        raise ValueError("El estado no contiene la clave 'question' en evaluation_task")

    question = state.get("question")
    if not question:
        raise ValueError("No se encontró 'question' en el estado en evaluation_task")

    answer = state.get("answer", "")
    if not answer:
        raise ValueError("No se encontró 'answer' en el estado en evaluation_task")

    chat_history = state.get("chat_history", [])
    config = kwargs.get('config', None)
    evaluation = await evaluate_answer(question, answer, chat_history, config)
    state["evaluation"] = evaluation
    logger.info(f"Evaluation task devuelve: {evaluation}")
    return state

@entrypoint(checkpointer=checkpointer)
async def rag_agent(state):
    if state is None or "question" not in state:
        raise ValueError("AGENTE: El estado inicial no contiene la clave 'question' en rag_agent")
    state = await retrieval_task(state)
    if state is None:
        raise ValueError("AGENTE: retrieval_task devolvió None")    
    state = await generation_task(state)
    if state is None:
        raise ValueError("AGENTE: generation_task devolvió None")
    answer = state.get("answer", "")
    if answer is None:
        raise ValueError("AGENTE: generation_task no devolvió 'answer'")
    state["final_answer"] = state["answer"]
    state = await evaluation_task(state)
    if state is None:
        raise ValueError("AGENTE: evaluation_task devolvió None")
    if "sí" in state.get("evaluation", "").lower():
        state["final_answer"] = state["answer"]
    else:
        state["final_answer"] = state["answer"] # "No se pudo generar una respuesta adecuada."
    return state

def retrieve_docs_sync(question: str):
    try:
        results = vectorstore.similarity_search_with_relevance_scores(
                question, k=5
        )
        return results or []
    except Exception as e:
        logger.error(f"Error en recuperación síncrona: {e}")
        return []

def generate_answer_sync(question: str, context: str, sources: str):
    prompt_text = response_prompt.format(context=context, sources=sources, question=question)
    messages = [{"role": "user", "content": prompt_text}]
    try:
        response_chunks = llm_chat(messages)
        full_response = ""
        for chunk in response_chunks:
            full_response = chunk
            logger.debug(f"Streaming respuesta síncrona: {full_response}")
        full_response_with_sources = f"{full_response}\n\nFuentes:\n{sources}"
        return full_response_with_sources
    except Exception as e:
        logger.error(f"Error en generación síncrona: {e}")
        return "Error al generar respuesta."

def evaluate_answer_sync(question: str, answer: str):
    prompt_text = eval_prompt.format(question=question, answer=answer)
    messages = [{"role": "user", "content": prompt_text}]
    try:
        response_chunks = llm_chat(messages)  # versión síncrona
        full_response = ""
        for chunk in response_chunks:
            full_response = chunk
            logger.info(f"Streaming evaluación síncrona: {full_response}")
        return full_response
    except Exception as e:
        logger.error(f"Error en evaluación síncrona: {e}")
        return "Error al evaluar respuesta."

@task
def retrieval_task_sync(state):
    logger.info(f"Retrieval task síncrona recibe estado: {state}")
    question = state.get("question")
    if not question:
        raise ValueError("No se encontró 'question' en el estado")
    docs = retrieve_docs_sync(question)
    logger.info(f"Documentos recuperados: {len(docs) if docs else 0}")
    if docs:
        context_text = "\n".join([doc.page_content for doc in docs])
        sources = [doc.metadata.get("source", "URL no disponible") for doc in docs]
        sources_text = "\n".join(sources)
    else:
        context_text = ""
        sources_text = ""
    state["context"] = context_text
    state["sources"] = sources_text
    logger.info(f"Retrieval task síncrona devuelve estado: {state}")
    return state

@task
def generation_task_sync(state):
    logger.info(f"Generation task síncrona recibe estado: {state}")
    if state is None:
        raise ValueError("Generation task recibió estado None")
    question = state.get("question")
    if not question:
        raise ValueError("No se encontró 'question' en el estado en generation_task_sync")
    context = state.get("context", "")
    sources = state.get("sources", "")
    answer = generate_answer_sync(question, context, sources)
    state["answer"] = answer
    logger.info(f"Generation task síncrona devuelve estado: {state}")
    return state

@task
def evaluation_task_sync(state):
    logger.info(f"Evaluation task síncrona recibe estado: {state}")
    if state is None or "question" not in state:
        raise ValueError("El estado no contiene la clave 'question' en evaluation_task_sync")
    question = state.get("question")
    if not question:
        raise ValueError("No se encontró 'question' en el estado en evaluation_task_sync")
    answer = state.get("answer", "")
    if not answer:
        raise ValueError("No se encontró 'answer' en el estado en evaluation_task_sync")
    evaluation = evaluate_answer_sync(question, answer)
    state["evaluation"] = evaluation
    logger.info(f"Evaluation task síncrona devuelve estado: {state}")
    return state

@entrypoint(checkpointer=checkpointer)
def rag_agent_sync(state):
    if state is None or "question" not in state:
        raise ValueError("AGENTE: El estado inicial no contiene la clave 'question' en rag_agent_sync")
    state = retrieval_task_sync(state)
    if state is None:
        raise ValueError("AGENTE: retrieval_task_sync devolvió None")
    state = generation_task_sync(state)
    if state is None:
        raise ValueError("AGENTE: generation_task_sync devolvió None")
    state = evaluation_task_sync(state)
    if state is None:
        raise ValueError("AGENTE: evaluation_task_sync devolvió None")
    if "sí" in state.get("evaluation", "").lower():
        state["final_answer"] = state["answer"]
    else:
        state["final_answer"] = "No se pudo generar una respuesta adecuada."
    return state

async def main():
    question = "¿Cómo puedo pedir la carrera horizontal?"
    config = {
        "configurable": {
            "thread_id": "thread-1",
            "checkpoint_ns": "default"
        }
    }

    async for event in rag_agent.astream(
        {"question": question},
        config=config,
        stream_mode=["updates", "messages"]
    ):
        if isinstance(event, dict):
            if "messages" in event:
                for msg in event["messages"]:
                    logger.debug(f"{msg['role']}: {msg['content']}")
            if "updates" in event:
                logger.debug(f"Estado actualizado: {event['updates']}")
            outputs = event.get("messages", []) or event.get("updates", {})
        elif isinstance(event, tuple):
            outputs = event[1]
            logger.debug(f"Estado actualizado: {outputs}")
        else:
            logger.debug(f"Evento: {event}")
            outputs = event

if __name__ == "__main__":
    asyncio.run(main())
