import os
import shutil
import logging
import requests
import json

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveJsonSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from backend.embedding_provider import EmbeddingProviderFactory

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)


# Leer constantes desde variables de entorno con valores por defecto
PERSIST_DIRECTORY = os.environ.get("PERSIST_DIRECTORY", "./.chroma")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "rag-sede")

def cargar_catalogo(crear_ficheros=False):
    catalogo_procedimientos = []
    ids_procedimientos = []

    urlbase = os.environ.get("URLBASE_CATALOGO")
    url = urlbase + "categoria/categorias?id_entidad=0&codigoIdioma=es"
    url_base_procedimientos = os.environ.get("URL_BASE_PROCEDIMIENTOS")

    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"Error al obtener categorías: {e}")
        return catalogo_procedimientos, ids_procedimientos

    json_categorias = response.json()
    if crear_ficheros:
        with open('categorias.json', 'w', encoding='utf-8') as outfile:
            outfile.write(response.text)

    for categoria in json_categorias:
        for subcategoria in categoria['subcategorias']['subcategorias']:
            url = (f"{urlbase}servicio/consultaCatalogoServicioPorSubCategoria"
                   f"?codigoIdioma=es&idCategoria={categoria['idCategoria']}"
                   f"&idSubCategoria={subcategoria['idSubcategoria']}&id_entidad=0")
            try:
                response = requests.get(url)
                response.raise_for_status()
            except Exception as e:
                logging.error(f"Error al obtener procedimientos: {e}")
                continue

            json_procedimientos = response.json()
            if crear_ficheros:
                with open(f'procedimientos_{categoria["idCategoria"]}_{subcategoria["idSubcategoria"]}.json', 'w', encoding='utf-8') as outfile:
                    outfile.write(response.text)

            for procedimiento in json_procedimientos:
                logging.info(f"{procedimiento['identificadorServicio']} - {procedimiento['denominacion']}")
                url = (f"{urlbase}servicio/consultaCatalogoServicioPorId"
                       f"?id_entidad=0&codigoIdioma=es&id_servicio={procedimiento['identificadorServicio']}")
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    if not response.text:
                        raise ValueError("Respuesta vacía")
                except Exception as e:
                    logging.error(f"Error al obtener detalle procedimiento: {e}")
                    continue

                json_procedimiento = response.json()
                if crear_ficheros:
                    with open('procedimientos.json', 'a', encoding='utf-8') as outfile:
                        outfile.write(response.text)

                metadata = {
                    "Procedimiento": json_procedimiento.get('denominacion', ''),
                    #"Departamento": json_procedimiento.get('departamento', ''),
                    "ID": json_procedimiento.get('identificadorServicio', ''),
                    "source": url_base_procedimientos + str(json_procedimiento.get('identificadorServicio', ''))
                }
                page_content = json_procedimiento
                catalogo_procedimientos.append((page_content, metadata))
                ids_procedimientos.append(procedimiento['identificadorServicio'])

    return catalogo_procedimientos, ids_procedimientos

def cargar_tramite(procs):
    catalogo_procedimientos = []
    urlbase = os.environ.get("URLBASE_CATALOGO")
    url_base_procedimientos = os.environ.get("URL_BASE_PROCEDIMIENTOS")
    
    for tramite_id in procs:
        try:
            url = (f"{urlbase}servicio/consultaCatalogoServicioPorId"
                   f"?id_entidad=0&codigoIdioma=es&id_servicio={tramite_id}")
            response = requests.get(url)
            response.raise_for_status()
            if not response.text:
                raise ValueError("Respuesta vacía")
        except Exception as e:
            logging.error(f"Error al obtener trámite {tramite_id}: {e}")
            return None

        json_tramite = response.json()
        metadata = {
            "Procedimiento": json_tramite.get('denominacion', ''),
            "Departamento": json_tramite.get('departamento', ''),
            "ID": json_tramite.get('identificadorServicio', ''),
            "source": url_base_procedimientos + str(json_tramite.get('identificadorServicio', ''))
        }
        page_content = json_tramite
        catalogo_procedimientos.append((page_content, metadata))
    return catalogo_procedimientos

def create_chroma_vectorstore(documents, embedding_provider, persist_directory, collection_name):
    # Borrar carpeta persistente para evitar duplicados y reiniciar índice
    if os.path.exists(persist_directory):
        try:
            shutil.rmtree(persist_directory)
            logging.info(f"Directorio {persist_directory} eliminado para reiniciar índice.")
        except Exception as e:
            logging.error(f"No se pudo eliminar el directorio {persist_directory}: {e}")
            raise

    try:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_provider.embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name,
            collection_metadata={"hnsw:space": "cosine"},
        )
        logging.info(f"Vectorstore creado con {len(documents)} documentos.")
        return vectorstore
    except Exception as e:
        logging.error(f"Error creando vectorstore: {e}")
        raise

def main():
    embedding_provider = EmbeddingProviderFactory.create()
    jsons, procs = cargar_catalogo()
    #procs = [3362, 2242, 657, 658, 5422]#, 5462, 654, 263, 1002, 765, 2023, 730, 101, 903]
    #jsons = cargar_tramite(procs) # Cargar un trámite específico para pruebas
    if not jsons:
        logging.error("No se cargaron documentos para indexar. Abortando.")
        return

    """
    urls = [os.environ.get("URL_BASE_PROCEDIMIENTOS") + str(id_proc) for id_proc in procs]
    loader = WebBaseLoader(urls, continue_on_failure=False)
    docs_list=loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=2000, chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents(docs_list)
    """
    json_splitter = RecursiveJsonSplitter(max_chunk_size=50000)

    docs_list = []
    for json_doc, metadata in jsons:
        json_chunks = json_splitter.split_json(json_doc, convert_lists=True)
        #json_chunks_compact = [json.dumps(chunk, separators=(',', ':')) for chunk in json_chunks]
        meta_list = [metadata] * len(json_chunks)
        docs = json_splitter.create_documents(texts=json_chunks, metadatas=meta_list)
        #docs = [Document(page_content=chunk, metadata=meta) for chunk, meta in zip(json_chunks_compact, meta_list)]
        docs_list.extend(docs)

    logging.info(f"Número de documentos originales: {len(jsons)}")
    logging.info(f"Número total de chunks generados: {len(docs_list)}")

    create_chroma_vectorstore(
        docs_list, embedding_provider, PERSIST_DIRECTORY, COLLECTION_NAME
    )

if __name__ == "__main__":
    main()
