# test_automatizado.py
"""
Módulo de Pruebas Automatizadas para el Sistema RAG
Sede Electrónica del Ayuntamiento de Murcia

Autor: David Albaladejo Serrano
Descripción: Sistema de evaluación automatizada para modelos de embeddings y LLM
"""

from contextlib import contextmanager
import os
import shutil
import time
import json
import asyncio
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from pathlib import Path

# Imports del sistema existente
from langchain_chroma import Chroma
from langchain.schema import Document
from .embedding_provider import EmbeddingProviderFactory
from .llm_provider import LLMProviderFactory
import backend.agentes as agentes_module  # Importar el módulo completo


# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

default_vectorstore = Chroma(
    collection_name=os.environ.get("COLLECTION_NAME", "rag-sede"),
    persist_directory=os.environ.get("PERSIST_DIRECTORY", "./.chroma"))

docs_dict = default_vectorstore.get()
all_documents = [
    Document(page_content=doc, metadata=meta)
    for doc, meta in zip(docs_dict['documents'], docs_dict['metadatas'])
]

default_llm = LLMProviderFactory.create().llm

@dataclass
class ModelConfig:
    """Configuración de modelo"""
    k:int # Número de documentos a recuperar por defecto
    nombre: str
    tipo: str  # 'embedding' o 'llm'
    parametros: Dict[str, Any]
    variables_entorno: Dict[str, str]

@dataclass
class ModelConfigHibrido:
    """Configuración para modelo híbrido"""
    nombre: str
    modelos_componentes: List[str]  # ['mxbai-embed-large', 'jina-embeddings-v2-base-es']
    estrategia: str  # 'union', 'rrf', 'weighted'
    pesos: Dict[str, float]
    parametros_fusion: Dict[str, Any]
    k_inicial: int = 20  # Cuántos docs recuperar de cada modelo

@dataclass
class PreguntaTest:
    """Preguntas de prueba con respuesta esperada"""
    id: int
    pregunta: str
    tramite_id: int
    tramite_nombre: str
    url_esperada: str
    categoria: str
    respuesta_esperada: str

@dataclass
class ResultadoEmbedding:
    """Resultado completo de evaluación de Embeddings con todas las métricas"""
    modelo: str
    pregunta_id: int
    metricas: Dict[str, float]
    latencia_ms: float
    score_max: float
    posicion_correcto: Optional[int]
    top_k_tramites: List[str]
    timestamp: str

@dataclass
class ResultadoLLM:
    """Resultado completo de evaluación de LLM con todas las métricas"""
    modelo: str
    pregunta_id: int
    contiene_url_correcta: bool
    score_autoevaluacion: float
    tiempo_respuesta_ms: float  
    faithfulness_score: float  
    bertscore_precision: float 
    bertscore_recall: float     
    bertscore_f1: float
    respuesta_completa: str
    evaluacion_feedback: str
    timestamp: str

class DatasetManager:
    """Gestor del dataset de pruebas"""
    
    def __init__(self, dataset_path: str = "./test_data"):
        self.dataset_path = Path(dataset_path)
        self.dataset_path.mkdir(exist_ok=True)
        
    def cargar_modelos_config(self) -> Tuple[List[ModelConfig], List[ModelConfig]]:
        """Carga configuraciones de modelos desde CSVs"""
        
        # Modelos de embedding
        embedding_csv = self.dataset_path / "embedding_models.csv"
        if not embedding_csv.exists():
            self._crear_csv_embedding_default(embedding_csv)
        
        df_emb = pd.read_csv(embedding_csv)
        modelos_embedding = []
        for _, row in df_emb.iterrows():
            config = ModelConfig(
                k=10,  # Valor por defecto, puede ser modificado
                nombre=row['nombre'],
                tipo='embedding',
                parametros=json.loads(row['parametros']) if 'parametros' in row else {},
                variables_entorno=json.loads(row['variables_entorno']) if 'variables_entorno' in row else {}
            )
            modelos_embedding.append(config)
        
        # Modelos LLM
        llm_csv = self.dataset_path / "llm_models.csv"
        if not llm_csv.exists():
            self._crear_csv_llm_default(llm_csv)
            
        df_llm = pd.read_csv(llm_csv)
        modelos_llm = []
        for _, row in df_llm.iterrows():
            config = ModelConfig(
                k=int(row['k_inicial']) if 'k_inicial' in row and not pd.isna(row['k_inicial']) else 0,
                nombre=row['nombre'],
                tipo='llm',
                parametros=json.loads(row['parametros']) if 'parametros' in row else {},
                variables_entorno=json.loads(row['variables_entorno']) if 'variables_entorno' in row else {},
            )
            modelos_llm.append(config)
            
        return modelos_embedding, modelos_llm
    
    def cargar_preguntas_test(self) -> List[PreguntaTest]:
        """Carga preguntas de prueba desde CSV"""
        preguntas_csv = self.dataset_path / "preguntas_test.csv"
        if not preguntas_csv.exists():
            self._crear_csv_preguntas_default(preguntas_csv)
            
        df = pd.read_csv(preguntas_csv)
        preguntas = []
        for _, row in df.iterrows():
            pregunta = PreguntaTest(
                id=row['id'],
                pregunta=row['pregunta'],
                respuesta_esperada= row.get('respuesta_esperada', ''),
                tramite_id=row['tramite_id'],
                tramite_nombre=row['tramite_nombre'],
                url_esperada=row['url_esperada'],
                categoria=row['categoria']
            )
            preguntas.append(pregunta)
            
        return preguntas

    def cargar_modelos_hibridos(self) -> List[ModelConfigHibrido]:
        """Carga configuraciones de modelos híbridos desde CSV"""
        hibridos_csv = self.dataset_path / "embedding_hibrid_models.csv"
        if not hibridos_csv.exists():
            self._crear_csv_hibridos_default(hibridos_csv)
        
        df_hibridos = pd.read_csv(hibridos_csv)
        modelos_hibridos = []
        
        for _, row in df_hibridos.iterrows():
            # Parsear listas y diccionarios desde strings
            logger.info(f"Cargando modelo híbrido: {row['nombre']}")
            modelos_componentes = json.loads(row['modelos_componentes'])
            logger.info(f"Modelos componentes: {modelos_componentes}")
            pesos = json.loads(row['pesos']) if isinstance(row['pesos'], str) and row['pesos'] else {}
            parametros_fusion = json.loads(row['parametros_fusion']) if isinstance(row['parametros_fusion'], str) and row['parametros_fusion'] else {}
            k_inicial = int(row['k_inicial']) if 'k_inicial' in row and not pd.isna(row['k_inicial']) else 20

            config = ModelConfigHibrido(
                nombre=row['nombre'],
                modelos_componentes=modelos_componentes,
                estrategia=row['estrategia'],
                pesos=pesos,
                parametros_fusion=parametros_fusion,
                k_inicial=k_inicial
            )
            modelos_hibridos.append(config)
        
        return modelos_hibridos

    def _crear_csv_hibridos_default(self, path: Path):
        """Crea un CSV de ejemplo para configuraciones híbridas"""
        data = [
            {
                'nombre': 'mxbai_jina_union',
                'modelos_componentes': '["mxbai-embed-large", "jina-embeddings-v2-base-es"]',
                'estrategia': 'union',
                'pesos': '{}',
                'parametros_fusion': '{}',
                'k_inicial': 20
            },
            {
                'nombre': 'mxbai_jina_rrf',
                'modelos_componentes': '["mxbai-embed-large", "jina-embeddings-v2-base-es"]',
                'estrategia': 'rrf',
                'pesos': '{}',
                'parametros_fusion': '{"c": 60}',
                'k_inicial': 20
            }
        ]
        pd.DataFrame(data).to_csv(path, index=False)
        logger.info(f"Creado CSV de ejemplo para modelos híbridos: {path}")

    def _crear_csv_embedding_default(self, path: Path):
        """Crea CSV de ejemplo para modelos de embedding"""
        data = [
            {
                'nombre': 'mxbai-embed-large',
                'parametros': '{"dimension": 1024}',
                'variables_entorno': '{"EMBEDDING_MODEL": "mxbai-embed-large", "EMBEDDING_PROVIDER": "ollama"}'
            },
            {
                'nombre': 'nomic-embed-text',
                'parametros': '{"dimension": 768}',
                'variables_entorno': '{"EMBEDDING_MODEL": "nomic-embed-text", "EMBEDDING_PROVIDER": "ollama"}'
            }
        ]
        pd.DataFrame(data).to_csv(path, index=False)
        logger.info(f"Creado CSV de ejemplo para embeddings: {path}")
    
    def _crear_csv_llm_default(self, path: Path):
        """Crea CSV de ejemplo para modelos LLM"""
        data = [
            {
                'nombre': 'gemma3:27b',
                'parametros': '{"temperature": 0.7, "max_tokens": 1500}',
                'variables_entorno': '{"LLM_MODEL": "gemma2:27b", "LLM_PROVIDER": "ollama"}'
            },
            {
                'nombre': 'mixtral:8x7b',
                'parametros': '{"temperature": 0.5, "max_tokens": 2000}',
                'variables_entorno': '{"LLM_MODEL": "mixtral:8x7b", "LLM_PROVIDER": "ollama"}'
            }
        ]
        pd.DataFrame(data).to_csv(path, index=False)
        logger.info(f"Creado CSV de ejemplo para LLMs: {path}")
    
    def _crear_csv_preguntas_default(self, path: Path):
        """Crea CSV de ejemplo para preguntas de prueba"""
        data = [
            {
                'id': 1,
                'pregunta': '¿Cómo puedo solicitar el padrón municipal?',
                'tramite_id': 1001,
                'tramite_nombre': 'Certificado de Empadronamiento',
                'url_esperada': 'https://sede.murcia.es/ficha-procedimiento?id=1001',
                'categoria': 'certificados'
            },
            {
                'id': 2,
                'pregunta': '¿Qué documentos necesito para el carnet de conducir?',
                'tramite_id': 2001,
                'tramite_nombre': 'Renovación Carnet de Conducir',
                'url_esperada': 'https://sede.murcia.es/ficha-procedimiento?id=2001',
                'categoria': 'documentacion'
            },
            {
                'id': 3,
                'pregunta': '¿Dónde puedo pagar las multas de tráfico?',
                'tramite_id': 3001,
                'tramite_nombre': 'Pago de Multas de Tráfico',
                'url_esperada': 'https://sede.murcia.es/ficha-procedimiento?id=3001',
                'categoria': 'pagos'
            },
            {
                'id': 4,
                'pregunta': '¿Cómo solicito ayudas sociales?',
                'tramite_id': 4001,
                'tramite_nombre': 'Solicitud de Ayudas Sociales',
                'url_esperada': 'https://sede.murcia.es/ficha-procedimiento?id=4001',
                'categoria': 'ayudas'
            },
            {
                'id': 5,
                'pregunta': '¿Qué necesito para registrar mi empresa?',
                'tramite_id': 5001,
                'tramite_nombre': 'Registro de Actividades Económicas',
                'url_esperada': 'https://sede.murcia.es/ficha-procedimiento?id=5001',
                'categoria': 'empresas'
            }
        ]
        pd.DataFrame(data).to_csv(path, index=False)
        logger.info(f"Creado CSV de ejemplo para preguntas: {path}")

class ModelSwitcher:
    """Gestor para cambiar modelos dinámicamente"""
    
    def __init__(self):
        self.original_env = dict(os.environ)
        self.temp_dirs = []
        self.original_vectorstore = None
    
    def cambiar_modelo(self, config: ModelConfig):
        """Cambia las variables de entorno para usar un modelo específico"""
        logger.info(f"Cambiando a modelo {config.tipo}: {config.nombre}")
        
        # Actualizar variables de entorno
        for key, value in config.variables_entorno.items():
            os.environ[key] = value
        
        # Recrear providers
        if config.tipo == 'embedding':
            # Guardar referencia original si es la primera vez
            if self.original_vectorstore is None:
                self.original_vectorstore = agentes_module.vectorstore
            
            # Crear directorio temporal único
            temp_dir = f"./.temp_{config.nombre}_{int(time.time())}"
            self.temp_dirs.append(temp_dir)
            
            # Recrear embedding provider
            global embedding_provider, embeddings
            embedding_provider = EmbeddingProviderFactory.create()
            embeddings = embedding_provider.embeddings

            test_embedding = embeddings.embed_query("test query")
            actual_dim = len(test_embedding)
            
            logger.info(f"Recreando vectorstore con {config.nombre}, de {actual_dim} dimensiones, para {len(all_documents)} documentos")
            
            if "nomic-embed-text" in config.nombre:
                # Añadir prefijo a cada documento
                docs_prefixed = []
                for doc in all_documents:
                    doc_prefixed = Document(
                        page_content=f"search_document: {doc.page_content}",
                        metadata=doc.metadata
                    )
                    docs_prefixed.append(doc_prefixed)
                documents_to_use = docs_prefixed
            else:
                documents_to_use = all_documents
            
            # Crear nuevo vectorstore con collection_name único
            new_vectorstore = Chroma.from_documents(
                documents=documents_to_use,
                embedding=embedding_provider.embeddings,
                collection_name=f"rag-sede-{config.nombre}-{int(time.time())}",  # Nombre único
                persist_directory=temp_dir,
                collection_metadata={"hnsw:space": "cosine"},
            )
            
            # CRÍTICO: Actualizar la variable global en el módulo agentes
            agentes_module.vectorstore = new_vectorstore
            
            # Verificar que el cambio fue efectivo
            logger.info(f"Vectorstore actualizado: {agentes_module.vectorstore._collection.name}")
            logger.info(f"Collection count: {agentes_module.vectorstore._collection.count()}")
            
        elif config.tipo == 'llm':
            global llm_provider, llm_chat, llm
            llm_provider = LLMProviderFactory.create(config.parametros.get('temperature', 0.5))
            llm_chat = llm_provider.chat
            llm = llm_provider.llm

            agentes_module.llm_provider = llm_provider
            agentes_module.llm = llm
            agentes_module.llm_chat = llm_chat
            logging.info(f"🔄 LLM actualizado en agentes: {agentes_module.llm_provider.__class__.__name__}")
    
    def restaurar_entorno(self):
        """Restaura las variables de entorno originales y limpia archivos temporales"""
        
        # Restaurar entorno
        os.environ.clear()
        os.environ.update(self.original_env)

        # Restaurar vectorstore original
        if self.original_vectorstore is not None:
            agentes_module.vectorstore = default_vectorstore
            
        # Limpiar directorios temporales
        for temp_dir in self.temp_dirs:
            if Path(temp_dir).exists():
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Directorio temporal eliminado: {temp_dir}")
                except Exception as e:
                    logger.warning(f"No se pudo eliminar {temp_dir}: {e}")
        
        self.temp_dirs.clear()
  
class EvaluadorEmbeddings:
    """Evaluador para modelos de embeddings"""
    def __init__(self, ks: List[int] = [3, 5, 10]):
        self.ks = sorted(ks)
        self.gestor_stores = None  # Para modelos híbridos
        self.fusionador = FusionadorResultados()  # Para fusión de resultados

    async def evaluar_modelo(self, config: Union[ModelConfig, ModelConfigHibrido], 
                            preguntas: List[PreguntaTest],
                            switcher: ModelSwitcher, 
                            configs_individuales: Optional[Dict[str, ModelConfig]] = None) -> List[ResultadoEmbedding]:
        """Evalúa un modelo de embedding individual o híbrido"""
        
        # Detectar tipo de configuración y delegar
        if isinstance(config, ModelConfigHibrido):
            if configs_individuales is None:
                raise ValueError("configs_individuales requerido para modelos híbridos")
            return await self._evaluar_modelo_hibrido(config_hibrido=config,configs_individuales=configs_individuales, preguntas=preguntas)
        else:
            return await self._evaluar_modelo_individual(config=config, preguntas=preguntas, switcher=switcher)
    
    async def _evaluar_modelo_individual(self, config: ModelConfig, preguntas: List[PreguntaTest],
                            switcher: ModelSwitcher) -> List[ResultadoEmbedding]:
        """Evalúa un modelo de embedding"""
        logger.info(f"Evaluando modelo embedding: {config.nombre}")
        
        # Cambiar al modelo específico
        switcher.cambiar_modelo(config)
        
        # VERIFICACIÓN: Confirmar que el vectorstore cambió
        import backend.agentes as agentes_module
        current_collection = agentes_module.vectorstore._collection.name
        logger.info(f"Usando collection: {current_collection}")
        
        # Verificación adicional: Hacer una consulta de prueba. 
        # Sirve además para cargar el modelo en memoria, en el caso de Ollama, para evitar latencias en la primera consulta.
        test_docs = await agentes_module.retrieve_docs("test query", [], False)
        if test_docs:
            logger.info(f"Test query score: {test_docs[0][1]:.6f}")
        
        resultados = []
        
        for pregunta in preguntas:
            try:
                # Medir tiempo de respuesta
                start_time = time.time()
                docs_scores = await agentes_module.retrieve_docs(pregunta.pregunta, [], False)
                latencia = (time.time() - start_time) * 1000  # en ms
                
                # LOG ADICIONAL: Verificar que los scores son diferentes
                if docs_scores:
                    primer_score = docs_scores[0][1]
                    logger.info(f"{config.nombre} - P{pregunta.id}: primer score = {primer_score:.6f}")
                
                # Calcula para cada k
                metricas = {}
                for k in self.ks:
                    prec, mrr, pos, top, smax = self._calcular_metricas(docs_scores,
                                                                        pregunta.tramite_id,
                                                                        k)
                    metricas[f"precision_at_{k}"] = prec
                    metricas[f"mrr_at_{k}"]        = mrr
                    # Solo guardamos la primera posición/top y score_max del mayor k
                    if k == max(self.ks):
                        posicion = pos
                        top_k_tramites = top
                        score_max = smax

                resultado = ResultadoEmbedding(
                    modelo=config.nombre,
                    pregunta_id=pregunta.id,
                    metricas=metricas,
                    latencia_ms=latencia,
                    score_max=score_max,
                    posicion_correcto=posicion,
                    top_k_tramites=top_k_tramites,
                    timestamp=datetime.now().isoformat()
                )
                
                resultados.append(resultado)
                logger.info(f"P{pregunta.id} con {config.nombre}: MRR={mrr:.3f}, Score={score_max:.6f}")
                
            except Exception as e:
                logger.error(f"Error evaluando pregunta {pregunta.id} con {config.nombre}: {e}")
        
        return resultados
    
    async def _evaluar_modelo_hibrido(self, 
                                     config_hibrido: ModelConfigHibrido,
                                     configs_individuales: Dict[str, ModelConfig],
                                     preguntas: List[PreguntaTest]) -> List[ResultadoEmbedding]:
        """Evalúa configuraciones híbridas"""
        
        # Inicializar vector stores una sola vez
        modelos_necesarios = []
        for modelo_name in config_hibrido.modelos_componentes:
            if modelo_name in configs_individuales:
                modelos_necesarios.append(configs_individuales[modelo_name])
        
        self.gestor_stores = GestorVectorStoresParalelos(modelos_necesarios)
        
        resultados = []
        
        for pregunta in preguntas:
            try:
                resultado = await self._evaluar_pregunta_hibrida(
                    config_hibrido, pregunta
                )
                resultados.append(resultado)
            except Exception as e:
                logger.error(f"Error evaluando pregunta {pregunta.id}: {e}")
        
        return resultados
    
    def _preparar_contexto(self, docs_scores: List[Tuple]) -> str:
        """Prepara el contexto a partir de documentos recuperados"""
        context_parts = []
        for doc, _ in docs_scores:
            context_parts.append(f"Procedimiento: {doc.metadata.get('Procedimiento', 'N/A')}")
            context_parts.append(f"Departamento: {doc.metadata.get('Departamento', 'N/A')}")
            context_parts.append(f"ID: {doc.metadata.get('ID', 'N/A')}")
            context_parts.append(f"source: {doc.metadata.get('source', 'N/A')}")
            context_parts.append(f"Contenido: {doc.page_content}")
            context_parts.append("FIN del Procedimiento")
            context_parts.append("---")
        return "\n".join(context_parts)

    async def _evaluar_pregunta_hibrida(self, 
                                      config: ModelConfigHibrido, 
                                      pregunta: PreguntaTest) -> ResultadoEmbedding:
        """Evalúa una pregunta con configuración híbrida"""
        
        start_time = time.time()
        
        # Búsquedas paralelas en todos los modelos componentes
        resultados_busqueda = await self.gestor_stores.buscar_paralelo(
            pregunta.pregunta, 
            config.modelos_componentes, 
            k=config.k_inicial
        )
        
        # Aplicar estrategia de fusión
        if config.estrategia == 'union':
            docs_fusionados = self.fusionador.fusion_union(resultados_busqueda, config.k_inicial)
        elif config.estrategia == 'rrf':
            docs_fusionados = self.fusionador.fusion_rrf(
                resultados_busqueda, config.k_inicial, 
                config.parametros_fusion.get('c', 60)
            )
        elif config.estrategia == 'rrf_z':
            docs_fusionados = self.fusionador.fusion_rrf_z(
                resultados_busqueda, config.k_inicial
            )
        elif config.estrategia == 'weighted':
            docs_fusionados = self.fusionador.fusion_weighted(
                resultados_busqueda, config.k_inicial, config.pesos
            )
        
        latencia = (time.time() - start_time) * 1000
        
        logger.info(f"{len(docs_fusionados)} documentos fusionados para P{pregunta.id} con {config.nombre} en {latencia:.2f} ms")

        escribir_contexto = False
        if escribir_contexto:
            # Preparar contexto para LLM
            contexto = self._preparar_contexto(docs_fusionados)
            with open(f"./test_data/contexto_{pregunta.id}.txt", "w", encoding="utf-8") as f:
                f.write(contexto)        

        # Calcular métricas para diferentes k
        metricas = {}
        for k in self.ks:
            precision, mrr, pos, top, score_max = self._calcular_metricas(
                docs_fusionados, pregunta.tramite_id, k
            )
            metricas[f"precision_at_{k}"] = precision
            metricas[f"mrr_at_{k}"] = mrr
        
        return ResultadoEmbedding(
            modelo=config.nombre,
            pregunta_id=pregunta.id,
            metricas=metricas,
            latencia_ms=latencia,
            score_max=score_max,
            posicion_correcto=pos,
            top_k_tramites=top,
            timestamp=datetime.now().isoformat()
        )

    def _calcular_metricas(self, docs_scores: List[Tuple], tramite_id_esperado: int, k: int) -> Tuple[float, float, Optional[int], List[str], float]:
        """Calcula métricas de evaluación"""
        if not docs_scores:
            return 0.0, 0.0, None, [], 0.0
            
        # Extraer información de los documentos
        tramites_encontrados = []
        scores = []
        posicion_correcto = None
        
        for i, (doc, score) in enumerate(docs_scores[:k]):
            tramite_id = doc.metadata.get('ID', -1)
            tramite_nombre = doc.metadata.get('Procedimiento', 'Desconocido')
            tramites_encontrados.append(f"{tramite_id}:{tramite_nombre}")
            scores.append(score)
            
            if tramite_id == tramite_id_esperado and posicion_correcto is None:
                posicion_correcto = i + 1
        
        # Calcular precisión@k
        precision = 1.0 if posicion_correcto is not None else 0.0
        
        # Calcular MRR
        mrr = 1.0 / posicion_correcto if posicion_correcto is not None else 0.0
        
        # Score máximo
        score_max = max(scores) if scores else 0.0

        return precision, mrr, posicion_correcto, tramites_encontrados, score_max

class FusionadorResultados:
    """Aplica diferentes estrategias de fusión"""
    
    @staticmethod
    def fusion_union(resultados: Dict[str, List[Tuple]], k: int) -> List[Tuple]:
        """Fusión UNION: combina y elimina duplicados"""
        todos_docs = []
        seen_ids = set()
        
        # Combinar todos los resultados
        for _, docs_scores in resultados.items():
            for doc, score in docs_scores:
                # Usar ID del documento o contenido inicial como identificador único
                doc_id = doc.metadata.get('ID', doc.page_content[:50])
                if doc_id not in seen_ids:
                    todos_docs.append((doc, score))
                    seen_ids.add(doc_id)
        
        # Ordenar por score descendente
        todos_docs.sort(key=lambda x: x[1], reverse=True)
        return todos_docs[:k]
    
    @staticmethod
    def fusion_rrf(resultados: Dict[str, List[Tuple]], k: int, c: int = 60) -> List[Tuple]:
        """Reciprocal Rank Fusion"""
        doc_scores = {}

        for _, docs_scores in resultados.items():
            for rank, (doc, score) in enumerate(docs_scores):
                doc_id = doc.metadata.get('ID', doc.page_content[:50])
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {'doc': doc, 'rrf_score': 0.0}
                
                # RRF score: 1 / (rank + c)
                doc_scores[doc_id]['rrf_score'] += 1.0 / (rank + 1 + c)
        
        # Ordenar por RRF score
        sorted_docs = sorted(
            doc_scores.values(), 
            key=lambda x: x['rrf_score'], 
            reverse=True
        )
        
        return [(item['doc'], item['rrf_score']) for item in sorted_docs[:k]]
    
    @staticmethod
    def fusion_rrf_z(resultados: Dict[str, List[Tuple]], k: int) -> List[Tuple]:
        """
        Fusión por suma de z-score: cada ranking se normaliza (z-score) 
        y se suman los z-scores de cada documento en todos los rankings.
        """
        doc_scores = {}

        # 1. Calcular z-score para cada ranking
        z_scores_by_model = {}
        for modelo, docs_scores in resultados.items():
            logger.info(f"Calculando z-scores para modelo {modelo} con {len(docs_scores)} documentos")
            scores = np.array([score for _, score in docs_scores])
            mu = scores.mean() if len(scores) > 0 else 0.0
            sigma = scores.std() if len(scores) > 0 else 1.0
            # Si la desviación estándar es 0 (todos los scores iguales), poner z-score a 0
            z_scores = [(doc, (score - mu) / sigma if sigma > 0 else 0.0) for doc, score in docs_scores]
            z_scores_by_model[modelo] = z_scores

        # 2. Sumar z-scores de cada documento
        for modelo, docs_zscores in z_scores_by_model.items():
            for doc, zscore in docs_zscores:
                doc_id = doc.metadata.get('ID', doc.page_content[:50])
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {'doc': doc, 'z_sum': 0.0}
                doc_scores[doc_id]['z_sum'] += zscore
        
        for doc_id, score_data in doc_scores.items():
            logger.info(f"Doc {doc_id} tiene z-score total: {score_data['z_sum']:.6f}")

        # 3. Ordenar documentos por suma de z-score descendente
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x['z_sum'],
            reverse=True
        )

        logger.info(f"Top {k} documentos por z-score:")
        for item in sorted_docs[:k]:  # Mostrar top 10 para depuración
            logger.info(f"Doc {item['doc'].metadata.get('ID', item['doc'].page_content[:50])} tiene z-score {item['z_sum']:.6f}")

        return [(item['doc'], item['z_sum']) for item in sorted_docs[:k]]
    
    @staticmethod
    def fusion_weighted(resultados: Dict[str, List[Tuple]], k: int, pesos: Dict[str, float]) -> List[Tuple]:
        """Fusión con pesos específicos por modelo"""
        doc_scores = {}
        
        for modelo, docs_scores in resultados.items():
            peso = pesos.get(modelo, 1.0)
            for doc, score in docs_scores:
                doc_id = doc.metadata.get('ID', doc.page_content[:50])
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {'doc': doc, 'weighted_score': 0.0}
                
                doc_scores[doc_id]['weighted_score'] += score * peso
        
        # Ordenar por score ponderado
        sorted_docs = sorted(
            doc_scores.values(), 
            key=lambda x: x['weighted_score'], 
            reverse=True
        )
        
        return [(item['doc'], item['weighted_score']) for item in sorted_docs[:k]]

class GestorVectorStoresParalelos:
    """Mantiene múltiples vector stores activos simultáneamente"""
    
    def __init__(self, modelos_embedding: List[ModelConfig]):
        self.vector_stores = {}
        self._inicializar_stores(modelos_embedding)
    
    @contextmanager
    def _aplicar_config_temporal(self, config):
        """Context manager para aplicar variables de entorno temporalmente"""
        env_backup = os.environ.copy()
        try:
            # Aplicar variables de entorno del modelo
            for key, value in config.variables_entorno.items():
                os.environ[key] = value
            yield
        finally:
            # Restaurar variables de entorno originales
            os.environ.clear()
            os.environ.update(env_backup)
    
    def _inicializar_stores(self, modelos_embedding: List[ModelConfig]):
        """Crea y mantiene todos los vector stores activos"""
        logger.info("Inicializando vector stores paralelos...")
        
        for config in modelos_embedding:
            logger.info(f"Creando vector store para {config.nombre}")
            
            # Crear embedding provider específico
            with self._aplicar_config_temporal(config):
                embedding_provider = EmbeddingProviderFactory.create()
            
            # Preparar documentos (con prefijo para nomic si es necesario)
            docs_to_use = self._preparar_documentos(config.nombre)
            
            # Crear vector store único
            vectorstore = Chroma.from_documents(
                documents=docs_to_use,
                embedding=embedding_provider.embeddings,
                collection_name=f"test-{config.nombre}-{int(time.time())}",
                persist_directory=f"./.temp_parallel_{config.nombre}",
                collection_metadata={"hnsw:space": "cosine"}
            )

            self.vector_stores[config.nombre] = vectorstore
            
            logger.info(f"✅ Vector store {config.nombre} inicializado: {vectorstore._collection.count()} docs")
    
    def _preparar_documentos(self, modelo_nombre: str) -> List[Document]:
        """Prepara documentos según el modelo (prefijo para nomic)"""
        if "nomic-embed-text" in modelo_nombre:
            return [
                Document(
                    page_content=f"search_document: {doc.page_content}",
                    metadata=doc.metadata
                ) for doc in all_documents
            ]
        return all_documents.copy()
    
    async def buscar_paralelo(self, query: str, modelos: List[str], k: int = 10) -> Dict[str, List[Tuple[Document, float]]]:
        """Realiza búsquedas paralelas en múltiples vector stores"""
        # Lanzar tareas asíncronas
        tasks = {
            modelo: asyncio.create_task(self._buscar_en_modelo(query, modelo, k))
            for modelo in modelos if modelo in self.vector_stores
        }
        resultados = {}
        for modelo, task in tasks.items():
            try:
                resultados[modelo] = await task  # List[Tuple[Document, float]]
            except Exception as e:
                logger.error(f"Error en búsqueda {modelo}: {e}")
                resultados[modelo] = []
        return resultados
    
    async def _buscar_en_modelo(self, query: str, modelo: str, k: int) -> List[Tuple[Document, float]] :
        """Búsqueda en un modelo específico"""
        vectorstore = self.vector_stores[modelo]
        results = await vectorstore.asimilarity_search_with_relevance_scores(
            query, k, 
        )
        return results

class EvaluadorLLM:
    """Evaluador para modelos LLM"""

    def __init__(self):
        self.gestor_stores  = None
    
    def _calcular_bertscore(self, respuesta_generada: str, respuesta_esperada: str) -> Dict[str, float]:
        """Calcula BERTScore entre respuesta generada y esperada"""
        try:
            import bert_score
            
            # Calcular BERTScore
            P, R, F1 = bert_score.score(
                [respuesta_generada], 
                [respuesta_esperada], 
                lang='es',  # Español para tu caso
                model_type='bert-base-multilingual-cased',  # Modelo multilingüe
                verbose=False
            )
            
            return {
                'bertscore_precision': float(P[0]),
                'bertscore_recall': float(R[0]), 
                'bertscore_f1': float(F1[0])
            }
            
        except Exception as e:
            logger.warning(f"Error calculando BERTScore: {e}")
            return {
                'bertscore_precision': 0.0,
                'bertscore_recall': 0.0,
                'bertscore_f1': 0.0
            }

    async def _calcular_faithfulness(self, pregunta: str, respuesta: str, contexto: str) -> float:
        """Calcula el score de faithfulness usando RAGAS 0.2.x"""
        try:
            from ragas.dataset_schema import SingleTurnSample
            from ragas.metrics import Faithfulness
            from ragas.llms import LangchainLLMWrapper
            
            # Preparar contexto
            retrieved_contexts = [contexto] if isinstance(contexto, str) else contexto
            
            # Crear muestra para evaluación
            sample = SingleTurnSample(
                user_input=pregunta,
                response=respuesta,
                retrieved_contexts=retrieved_contexts
            )

            logging.info(f"🧮 Calculando faithfulness para pregunta: {pregunta}")
            logging.info(f"🧮 Con respuesta: {respuesta}")
            logging.info(f"Tipo de agentes_module.llm: {type(default_llm)}")
            logging.info(f"Nombre del tipo de llm: {type(default_llm).__name__}")
            
            evaluator_llm = LangchainLLMWrapper(default_llm)
            
            # Crear scorer con API v0.2
            scorer = Faithfulness(llm=evaluator_llm)
            faithfulness_score = await scorer.single_turn_ascore(sample)
            
            return float(faithfulness_score) if faithfulness_score is not None else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculando faithfulness: {e}")
            return 0.0
    
    async def evaluar_modelo(self, config: ModelConfig, preguntas: List[PreguntaTest], 
                           switcher: ModelSwitcher) -> List[ResultadoLLM]:
        """Evalúa un modelo LLM"""
        logger.info(f"Evaluando modelo LLM: {config.nombre}")
        
        # Cambiar al modelo específico
        switcher.cambiar_modelo(config)
        resultados = []
        

        respuesta_completa = ""
        async for chunk in agentes_module.generate_answer("test", "test", []):
            respuesta_completa += chunk.content if hasattr(chunk, 'content') else chunk
        
        for pregunta in preguntas:
            try:
                context = self._preparar_contexto(pregunta.id, config.k)
                
                # Generar respuesta midiendo tiempo
                start_time = time.time()
                respuesta_completa = ""
                async for chunk in agentes_module.generate_answer(pregunta.pregunta, context, []):
                    respuesta_completa += chunk.content if hasattr(chunk, 'content') else chunk
                tiempo_respuesta = (time.time() - start_time) * 1000  # en ms
                
                # Evaluar respuesta
                contiene_url = pregunta.url_esperada in respuesta_completa
                
                # Autoevaluación
                evaluacion = await agentes_module.evaluate_answer(pregunta.pregunta, respuesta_completa, [])
                score_auto = self._extraer_score_evaluacion(evaluacion)

                # Calcular faithfulness
                faithfulness_score = 0.0
                faithfulness_score = await self._calcular_faithfulness(
                               pregunta.pregunta, respuesta_completa, context
                )
                
                # Calcular BERTScore si hay respuesta esperada
                bertscore_metrics = {}
                if hasattr(pregunta, 'respuesta_esperada') and pregunta.respuesta_esperada:
                    bertscore_metrics = self._calcular_bertscore(
                        respuesta_completa, pregunta.respuesta_esperada
                    )
                    
                resultado = ResultadoLLM(
                    modelo=config.nombre,
                    pregunta_id=pregunta.id,
                    contiene_url_correcta=contiene_url,
                    score_autoevaluacion=score_auto,
                    tiempo_respuesta_ms=tiempo_respuesta,
                    respuesta_completa=respuesta_completa,
                    evaluacion_feedback=evaluacion,
                    faithfulness_score=faithfulness_score,
                    bertscore_precision=bertscore_metrics.get('bertscore_precision', 0.0),
                    bertscore_recall=bertscore_metrics.get('bertscore_recall', 0.0),
                    bertscore_f1=bertscore_metrics.get('bertscore_f1', 0.0),
                    timestamp=datetime.now().isoformat()
                )
                resultados.append(resultado)
                
                logger.debug(f"P{pregunta.id}: URL={contiene_url}, Score={score_auto:.1f}, Tiempo={tiempo_respuesta:.1f}ms")
                
            except Exception as e:
                logger.error(f"Error evaluando pregunta {pregunta.id} con LLM: {e}")
                
        return resultados
    
    def _preparar_contexto(self, id_pregunta: int, k:int) -> str:
        """Prepara el contexto a partir de documentos recuperados"""
        with open(f"./test_data/contexto_k{k}/contexto_{id_pregunta}.txt", "r", encoding="utf-8") as f:
            contexto = f.read()
        if not contexto:
            logger.warning(f"No se encontró contexto para P{id_pregunta}, usando vacío")
            return "No se encontró contexto relevante."
        return contexto
    
    def _extraer_score_evaluacion(self, evaluacion: str) -> float:
        """Extrae un score numérico de la evaluación textual"""
        if "sí" in evaluacion.lower():
            return 1.0
        elif "no" in evaluacion.lower():
            return 0.0
        else:
            return 0.5  # Score neutro si no es claro

class GeneradorReportes:
    """Generador de reportes de evaluación"""
    
    def __init__(self, output_dir: str = "./test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generar_reporte_embeddings(self, resultados: List[ResultadoEmbedding]) -> str:
        # Construir un DataFrame plano con cada métrica
        rows = []
        for r in resultados:
            base = {
                'modelo':            r.modelo,
                'pregunta_id':       r.pregunta_id,
                'latencia_ms':       r.latencia_ms,
                'score_max':         r.score_max
            }
            rows.append({**base, **r.metricas})

        df = pd.DataFrame(rows)

        # === Resumen por modelo ===
        agg_dict = {'latencia_ms': ['mean', 'std'], 'score_max': ['mean', 'std']}
        for k in [3, 5, 10]:
            agg_dict[f'precision_at_{k}'] = ['mean', 'std']
            agg_dict[f'mrr_at_{k}']       = ['mean', 'std']

        metricas_por_modelo = df.groupby('modelo').agg(agg_dict).round(3)
        
        # Guardar resultados detallados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archivo_detallado = self.output_dir / f"embeddings_detallado_{timestamp}.csv"
        df.to_csv(archivo_detallado, index=False)
        
        # Guardar resumen
        archivo_resumen = self.output_dir / f"embeddings_resumen_{timestamp}.csv"
        metricas_por_modelo.to_csv(archivo_resumen, index_label='modelo')
        
        logger.info(f"Reporte embeddings guardado: {archivo_resumen}")
        return str(archivo_resumen)
    
    def generar_reporte_llm(self, resultados: List[ResultadoLLM]) -> str:
        """Genera reporte para evaluación de LLM"""
        df = pd.DataFrame([asdict(r) for r in resultados])
        
        # Calcular métricas agregadas por modelo
        metricas_por_modelo = df.groupby('modelo').agg({
            'contiene_url_correcta': 'mean',
            'score_autoevaluacion': ['mean', 'std'],
            'tiempo_respuesta_ms': ['mean', 'std'],
            'faithfulness_score': ['mean', 'std'],
            'bertscore_precision': ['mean', 'std'],
            'bertscore_recall': ['mean', 'std'],
            'bertscore_f1': ['mean', 'std']
        }).round(3)
        
        # Guardar resultados detallados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archivo_detallado = self.output_dir / f"llm_detallado_{timestamp}.csv"
        df.to_csv(archivo_detallado, index=False)
        
        # Guardar resumen
        archivo_resumen = self.output_dir / f"llm_resumen_{timestamp}.csv"
        df_resumen = metricas_por_modelo.reset_index()
        df_resumen.to_csv(archivo_resumen, index=False)
        
        logger.info(f"Reporte LLM guardado: {archivo_resumen}")
        return str(archivo_resumen)
    
    def generar_reporte_comparativo(self, resultados_emb: List[ResultadoEmbedding], 
                                  resultados_llm: List[ResultadoLLM]) -> str:
        """Genera reporte comparativo combinado"""
        # Crear resumen ejecutivo
        logger.info("Generando reporte comparativo de modelos de embeddings y LLM")

        # Construir un DataFrame plano con cada métrica
        rows = []
        for r in resultados_emb:
            base = {
                'modelo':            r.modelo,
                'pregunta_id':       r.pregunta_id,
                'latencia_ms':       r.latencia_ms,
                'score_max':         r.score_max
            }
            rows.append({**base, **r.metricas})

        df_emb = pd.DataFrame(rows)

        logger.info(f"Total de resultados de embeddings: {len(df_emb)}")

        # === Resumen por modelo ===
        agg_dict = {'latencia_ms': 'mean', 'score_max': 'mean'}
        for k in [3, 5, 10]:
            agg_dict[f'precision_at_{k}'] = 'mean'
            agg_dict[f'mrr_at_{k}']       = 'mean'

        emb_summary = df_emb.groupby('modelo').agg(agg_dict).round(3)

        logger.info(f"Resumen de embeddings generado:")
        logger.info(f"\n{emb_summary}")

        # Métricas de LLM
        df_llm = pd.DataFrame([asdict(r) for r in resultados_llm])
        llm_summary = df_llm.groupby('modelo').agg({
            'contiene_url_correcta': 'mean',
            'score_autoevaluacion': 'mean',
            'tiempo_respuesta_ms': 'mean',
            'faithfulness_score': 'mean',
            'bertscore_precision': 'mean',
            'bertscore_recall': 'mean',
            'bertscore_f1': 'mean'
        }).round(3)

        logger.info(f"Total de resultados de LLM: {len(df_llm)}")
        logger.info(f"Resumen de LLM generado:")
        logger.info(f"\n{llm_summary}")
        
        # Crear reporte combinado
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reporte_path = self.output_dir / f"reporte_comparativo_{timestamp}.md"
        
        logger.info(f"Guardando reporte comparativo en: {reporte_path}")
        with open(reporte_path, 'w', encoding='utf-8') as f:
            f.write(f"# Reporte de Evaluación de Modelos\n")
            f.write(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Resumen de Modelos de Embeddings\n\n")
            f.write(emb_summary.to_markdown())
            f.write("\n\n")
            
            f.write("## Resumen de Modelos LLM\n\n")
            f.write(llm_summary.to_markdown())
            f.write("\n\n")
            
            # Mejores modelos
            mejor_embedding_idx = emb_summary['precision_at_10'].idxmax()
            mejor_embedding = emb_summary.loc[mejor_embedding_idx]
            mejor_llm_idx = llm_summary['faithfulness_score'].idxmax()
            mejor_llm = llm_summary.loc[mejor_llm_idx]
            
            f.write("## Recomendaciones\n\n")
            f.write(f"**Mejor modelo de embedding:** {mejor_embedding_idx} ")
            f.write(f"(Precisión@10: {mejor_embedding['precision_at_10']:.3f})\n\n")
            f.write(f"**Mejor modelo LLM:** {mejor_llm_idx} ")
            f.write(f"(Faithfulness Score: {mejor_llm['faithfulness_score']:.3f})\n\n")
        
        logger.info(f"Reporte comparativo guardado: {reporte_path}")
        return str(reporte_path)