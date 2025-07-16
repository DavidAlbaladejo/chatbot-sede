import datetime
import logging
import streamlit as st
import asyncio
from uuid import uuid4
from dotenv import load_dotenv
from backend.agentes import rag_agent
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)

def registrar_feedback(session_id):
    feedback_valor = st.session_state.get(f"feedback_{session_id}")
    logger.info(f"Feedback recabado para sesión {session_id}: {feedback_valor}")

    if feedback_valor is not None:
        st.session_state.feedback_data.append({
            "session_id": session_id,
            "valoracion": feedback_valor,
            "timestamp": datetime.datetime.now()
        })
        logger.info(f"Feedback registrado: {feedback_valor}")
    logger.info(f"Feedback actualizado")

async def process_user_input_with_streaming(user_input, response_container):
    """Procesa la entrada con streaming real y devuelve el estado final"""
    
    # Crear el estado inicial para el agente
    state = {
        "question": user_input,
        "chat_history": [{"role": msg.type, "content": msg.content} for msg in history.messages[:-1]]
    }
    
    config = {
        "configurable": {
            "thread_id": st.session_state.session_id,
            "checkpoint_ns": "default"
        }
    }
    
    full_response = ""
    final_state = None
    
    try:   
        # Mostrar indicador de procesamiento inicial
        response_container.info("🤖 Procesando tu consulta...")
        
        # Streaming con astream
        async for event in rag_agent.astream(
            state, 
            config=config, 
            stream_mode=["messages", "updates"]
        ):
            # El evento es una tupla: (event_type, data)
            if isinstance(event, tuple) and len(event) == 2:
                event_type, event_data = event
                #logger.info(f"Tipo de datos del evento: {type(event_data)}")
                #logger.info(f"Datos del evento: {event_data}")
                if event_type == "updates":
                    # event_data es un diccionario con las actualizaciones de cada tarea
                    for task_name, task_output in event_data.items():
                        
                        if task_name == "retrieval_task":
                            response_container.info("🔍 Documentos encontrados y analizados...")
                            logger.info(f"Documentos encontrados")
                            
                        elif task_name == "generation_task":
                            response_container.info("✍️ Generando respuesta...")
                            logger.info(f"Generando respuesta")
                            
                            # Mostrar la respuesta conforme se genera
                            if isinstance(task_output, dict) and "answer" in task_output:
                                answer = task_output["answer"]
                                response_container.markdown(answer + "▌")
                                full_response = answer
                                
                        elif task_name == "evaluation_task":
                            logger.info(f"Evaluando respuesta")
                            response_container.info("🔍 Evaluando calidad de la respuesta...")
                            
                        # Capturar el estado final del agente
                        elif task_name == "__end__" or "final_answer" in task_output:
                            final_state = task_output
                elif event_type == "messages":
                    message_chunk, metadata = event_data
                    # Filtrar solo mensajes del nodo de generación
                    if (metadata.get('langgraph_node') == 'generation_task' and 
                        hasattr(message_chunk, 'content') and message_chunk.content):
                        response_container.info("✍️ Generando respuesta...")
                        full_response += message_chunk.content
                        # Mostrar respuesta con cursor de escritura
                        response_container.markdown(full_response + "▌")
                    
        # Mostrar respuesta final sin cursor
        if full_response:
            response_container.markdown(full_response)
        
        # Si no obtuvimos el estado final completo, hacer una llamada adicional
        if not final_state or "evaluation" not in final_state:
            final_state = await rag_agent.ainvoke(state, config=config)
        
        return final_state, full_response
        
    except Exception as e:
        logger.error(f"Error en streaming: {e}")
        response_container.error(f"Error: {e}")
        return None, ""
    
def run_async_in_streamlit(coro):
    """Helper para ejecutar código asíncrono en Streamlit"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)

# Configuración inicial de la página
st.set_page_config(
    page_title="Chatbot Sede Murcia",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_custom_styles():
    """Aplica los estilos CSS desde el archivo externo custom.css"""
    try:
        with open('static/css/custom.css', 'r', encoding='utf-8') as f:
            css_content = f.read()
        
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("No se pudo encontrar el archivo custom.css")
    except Exception as e:
        st.error(f"Error al cargar los estilos: {e}")

#apply_custom_styles()

#Logo de la aplicación
st.logo("static/assets/logo.png", size="large")

# Cabecera personalizada
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 2rem;">
    <img src="app/static/assets/logo.png" width="400" style="margin-right: 1rem;">
</div>
<div style="display: flex; align-items: center; margin-bottom: 2rem;">
    <h1 style="margin: 0;">Chatbot de la Sede Electrónica de Murcia</h1>
</div>
""", unsafe_allow_html=True)

# Gestión del estado de sesión
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
    st.session_state.feedback_data = []

# Historial de mensajes
history = StreamlitChatMessageHistory(key="chat_history")
if not history.messages:
    history.add_ai_message("¡Bienvenido! Soy tu asistente virtual para trámites del Ayuntamiento de Murcia. ¿En qué puedo ayudarte?")

# Mostrar historial con estilos
for msg in history.messages:
    avatar = "👤" if isinstance(msg, HumanMessage) else "🏛️"
    with st.chat_message(msg.type, avatar=avatar):
        st.markdown(msg.content)
        if hasattr(msg, 'metadata') and msg.metadata.get('evaluation'):
            with st.expander("Evaluación del sistema"):
                st.write(msg.metadata['evaluation'])

# Procesar entrada del usuario
if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    history.add_user_message(prompt)
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🏛️"):
        with st.spinner("Buscando información..."):
            response_container = st.empty()
            try:
                final_state, full_response = run_async_in_streamlit(process_user_input_with_streaming(
                    prompt,
                    response_container
                ))
                
                if final_state:
                    full_response = final_state.get("final_answer", "No se pudo generar una respuesta.")
                    evaluation = final_state.get("evaluation", "No se pudo generar una evaluación.")

                    if evaluation:
                        with st.expander("Evaluación interna"):
                            st.write(evaluation)

                    # Sistema de feedback
                    col1, col2 = st.columns([4, 1])
                    with col2:
                        st.feedback("thumbs",key=f"feedback_{st.session_state.session_id}",
                                    on_change=registrar_feedback,args=(st.session_state.session_id,))

                    # Agregar respuesta al historial
                    history.add_ai_message(AIMessage(
                        content=full_response,
                        metadata={"evaluation": evaluation}
                    ))

            except Exception as e:
                st.error(f"Error en el procesamiento: {e}")
                logger.error(f"Error en chat principal: {e}")


# Panel lateral con estadísticas
with st.sidebar:
    st.header("Configuración y Métricas")
    
    if st.button("🔄 Nueva Conversación"):
        history.clear()
        st.session_state.session_id = str(uuid4())
        history.add_ai_message("¡Bienvenido! Soy tu asistente virtual para trámites del Ayuntamiento de Murcia. ¿En qué puedo ayudarte?")
        st.rerun()
    
    with st.expander("📊 Estadísticas de Uso"):
        if st.session_state.feedback_data:
            positivo = sum(fb["valoracion"] for fb in st.session_state.feedback_data) / len(st.session_state.feedback_data)
            st.metric("Valoraciones Positivas", f"{positivo*5}/5")
            st.progress(positivo)
        else:
            st.write("No hay datos de feedback aún")
    

# Footer
st.markdown("""
---
<div class="footer">
    <p>💡 <strong>Consejo:</strong> Se especifico en tus preguntas para obtener mejores respuestas</p>
    <p>🔒 Tus conversaciones son privadas y seguras</p>
    <p>🏛️ Todas las respuestas son generadas por IA y no son vinculantes ni están garantizadas</p>
    <p>📞 En caso de duda, contacte con el servicio de antención al ciudadano del Ayuntamiento de Murcia (<a href="tel:010">📞010</a>) o consulte la <a href="https://sede.murcia.es">🌐Sede Electrónica</a></p>
</div>
""", unsafe_allow_html=True)
