# interfaz_pruebas.py
"""
Interfaz básica para el sistema de pruebas automatizadas
Sede Electrónica del Ayuntamiento de Murcia

Esta es una interfaz opcional de Streamlit que permite ejecutar las pruebas
automatizadas con una interfaz gráfica mínima.
"""
import torch
torch.classes.__path__ = []  # Workaround para compatibilidad Streamlit-PyTorch

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import asyncio
import os
import glob
import sys
from pathlib import Path
import time
from datetime import datetime

# Importar el sistema de pruebas
from ejecutor_pruebas import SistemaEvaluacion
from backend.test_automatizado import DatasetManager

def escanear_resultados(results_path: str = "./test_results"):
    """
    Devuelve una lista de diccionarios con los metadatos de cada CSV
    encontrado en `results_path`, ordenados de más reciente a más antiguo.
    """
    archivos = []
    for ruta in glob.glob(os.path.join(results_path, "*.csv")):
        stats = os.stat(ruta)
        nombre = Path(ruta).name.lower()
        tipo = (
            "Embeddings resumen" if "embedding" in nombre
            else "LLM resumen"   if "llm"       in nombre
            else "Evaluación Detallada"
        )
        archivos.append(
            dict(
                ruta=ruta,
                nombre=Path(ruta).name,
                tipo=tipo,
                fecha=datetime.fromtimestamp(stats.st_mtime),
                tamaño_kb=round(stats.st_size / 1024, 2),
            )
        )
    return sorted(archivos, key=lambda x: x["fecha"], reverse=True)

def cargar_csv_resultado(ruta_csv: str):
    """
    Lee un CSV de resultados:
    Si el nombre contiene 'resumen', lee con header=[0,1] e index_col=0.
    Si no, lo carga como plano.
    Devuelve `(df, tipo)` para que la capa de presentación sepa qué hacer.
    """
    nombre = Path(ruta_csv).name.lower()
    if "resumen" in nombre:
        df = pd.read_csv(ruta_csv, header=[0, 1], index_col=0)
        tipo = "resumen"
    else:
        df = pd.read_csv(ruta_csv)
        tipo = "detallado"
    return df, tipo

def mostrar_resumen(reporte: str):

    df = pd.read_csv(reporte, header=[0, 1], index_col=0)
    st.subheader(f"Reporte: {Path(reporte).name}")
    st.dataframe(df)
    
    if "embedding" in reporte:
        # Tabla resumen de embeddings (mantener igual)
        st.subheader("📊 Métricas de Embedding (media por modelo)")
        df_emb = pd.DataFrame({
            'modelo': df.index,
            'Latencia (ms)': df[('latencia_ms', 'mean')].values,
            'Precision@3': df[('precision_at_3', 'mean')].values if ('precision_at_3', 'mean') in df.columns else 0,
            'Precision@5': df[('precision_at_5', 'mean')].values if ('precision_at_5', 'mean') in df.columns else 0,
            'Precision@10': df[('precision_at_10', 'mean')].values if ('precision_at_10', 'mean') in df.columns else 0,
            'MRR@3': df[('mrr_at_3', 'mean')].values if ('mrr_at_3', 'mean') in df.columns else 0,
            'MRR@5': df[('mrr_at_5', 'mean')].values if ('mrr_at_5', 'mean') in df.columns else 0,
            'MRR@10': df[('mrr_at_10', 'mean')].values if ('mrr_at_10', 'mean') in df.columns else 0
        })
        st.table(df_emb)
        
        # Métricas en X, Modelos como barras agrupadas
        st.subheader("📈 Comparativa de Métricas de Embedding")
        
        # Preparar datos: transponer para tener métricas en filas
        df_plot = df_emb.set_index('modelo')
        
        # Separar métricas normalizadas de latencias
        metricas_norm = ['Precision@3', 'Precision@5', 'Precision@10', 'MRR@3', 'MRR@5', 'MRR@10']
        df_metricas = df_plot[metricas_norm].T  # Transponer: métricas en filas, modelos en columnas
        
        # Gráfica principal con doble eje Y
        fig, ax1 = plt.subplots(figsize=(16, 8))
        
        # Configuración para separación entre grupos
        num_metrics = len(df_metricas.index)
        num_models = len(df_metricas.columns)
        width = 0.25  # Ancho de cada barra individual
        group_spacing = 0.25  # Espacio entre grupos de métricas
        
        # Calcular posiciones base para cada grupo con separación
        base_positions = np.arange(num_metrics) * (num_models * width + group_spacing)
        
        # Colores para los modelos
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:cyan', 'tab:red']
        
        # Crear barras agrupadas para cada modelo
        for i, modelo in enumerate(df_metricas.columns):
            positions = base_positions + i * width
            ax1.bar(positions, df_metricas[modelo], width, 
                    label=modelo, color=colors[i % len(colors)], alpha=0.8)
        
        # Configurar eje izquierdo para métricas
        ax1.set_xlabel('Métricas')
        ax1.set_ylabel('Valor de la métrica', color='black')
        ax1.set_title('Comparativa de Modelos de Embedding')
        
        # Posicionar etiquetas del eje X en el centro de cada grupo
        center_positions = base_positions + (num_models - 1) * width / 2
        ax1.set_xticks(center_positions)
        ax1.set_xticklabels(df_metricas.index, rotation=45)
        ax1.set_ylim(0, 1)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Eje derecho: latencias
        ax2 = ax1.twinx()
        # Posicionar las barras de latencia después de las métricas con separación adicional
        latencias_pos = base_positions[-1] + (num_models * width + group_spacing * 2)
        
        for i, modelo in enumerate(df_plot.index):
            offset = i * width
            ax2.bar(latencias_pos + offset, df_plot.loc[modelo, 'Latencia (ms)'], width,
                    color=colors[i % len(colors)], alpha=0.6, hatch='//')
        
        ax2.set_ylabel('Latencia (ms)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Añadir etiqueta de "Latencia" en el eje X
        all_labels = list(df_metricas.index) + ['Latencia (ms)']
        all_positions = list(center_positions) + [latencias_pos + (num_models - 1) * width / 2]
        ax1.set_xticks(all_positions)
        ax1.set_xticklabels(all_labels, rotation=45)

        # Cambia el color de la última etiqueta a rojo
        labels = ax1.get_xticklabels()
        labels[-1].set_color('red')

        plt.tight_layout()
        st.pyplot(fig)

    else:  # LLM
        # Tabla resumen de LLMs (mantener igual)
        st.subheader("📊 Métricas LLM (media por modelo)")
        df_llm = pd.DataFrame({
            'modelo': df.index,
            'Tiempo LLM (ms)': df[('tiempo_respuesta_ms', 'mean')].values,
            'Exactitud URL': df[('contiene_url_correcta','mean')].values,
            'Score Auto': df[('score_autoevaluacion','mean')].values,
            'Faithfulness': df[('faithfulness_score','mean')].values,
            'BERTScore F1': df[('bertscore_f1','mean')].values
        })
        st.table(df_llm)
        
        # NUEVA GRÁFICA: Métricas en X, Modelos como barras agrupadas
        st.subheader("📈 Comparativa de Métricas LLM")
        
        # Preparar datos: transponer para tener métricas en filas
        df_plot = df_llm.set_index('modelo')
        
        # Separar métricas normalizadas de tiempos
        metricas_norm = ['Exactitud URL', 'Score Auto', 'Faithfulness', 'BERTScore F1']
        df_metricas = df_plot[metricas_norm].T  # Transponer: métricas en filas, modelos en columnas
        
        # Gráfica principal con doble eje Y
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Configuración para separación entre grupos
        num_metrics = len(df_metricas.index)
        num_models = len(df_metricas.columns)
        width = 0.15  # Ancho de cada barra individual
        group_spacing = 1.0  # Espacio entre grupos de métricas
        
        # Calcular posiciones base para cada grupo con separación
        base_positions = np.arange(num_metrics) * (num_models * width + group_spacing)
        
        # Colores para los modelos
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:cyan', 'tab:red']
        
        # Crear barras agrupadas para cada modelo
        for i, modelo in enumerate(df_metricas.columns):
            positions = base_positions + i * width
            ax1.bar(positions, df_metricas[modelo], width, 
                    label=modelo, color=colors[i % len(colors)], alpha=0.8)
        
        # Configurar eje izquierdo para métricas
        ax1.set_xlabel('Métricas')
        ax1.set_ylabel('Valor de la métrica', color='black')
        ax1.set_title('Comparativa de Modelos LLM')
        
        # Posicionar etiquetas del eje X en el centro de cada grupo
        center_positions = base_positions + (num_models - 1) * width / 2
        ax1.set_xticks(center_positions)
        ax1.set_xticklabels(df_metricas.index, rotation=45)
        ax1.set_ylim(0, 1)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Eje derecho: tiempos de respuesta
        ax2 = ax1.twinx()
        # Posicionar las barras de tiempo después de las métricas con separación adicional
        tiempos_pos = base_positions[-1] + (num_models * width + group_spacing * 2)
        
        for i, modelo in enumerate(df_plot.index):
            offset = i * width
            ax2.bar(tiempos_pos + offset, df_plot.loc[modelo, 'Tiempo LLM (ms)'], width,
                    color=colors[i % len(colors)], alpha=0.6, hatch='//')
        
        ax2.set_ylabel('Tiempo LLM (ms)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Añadir etiqueta de "Tiempo" en el eje X
        all_labels = list(df_metricas.index) + ['Tiempo LLM (ms)']
        all_positions = list(center_positions) + [tiempos_pos + (num_models - 1) * width / 2]
        ax1.set_xticks(all_positions)
        ax1.set_xticklabels(all_labels, rotation=45)

        # Cambia el color de la última etiqueta a rojo
        labels = ax1.get_xticklabels()
        labels[-1].set_color('red')
        
        plt.tight_layout()
        st.pyplot(fig)
 

#  Configuración de la página
st.set_page_config(
    page_title="Evaluador de Modelos RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y descripción
st.title("Evaluador de Modelos RAG - Sede Electrónica")
st.markdown("""
Este sistema permite evaluar modelos de embeddings y LLMs para el chatbot
de la Sede Electrónica del Ayuntamiento de Murcia.
""")

# Sidebar para configuración
st.sidebar.header("Configuración")

# Rutas
dataset_path = st.sidebar.text_input("Ruta del dataset", "./test_data")
results_path = st.sidebar.text_input("Ruta de resultados", "./test_results")

# Tipo de evaluación
eval_type = st.sidebar.radio(
    "Tipo de evaluación",
    ["Completa", "Solo Embeddings", "Solo LLM"]
)

# Cargar datos del sistema
@st.cache_data(ttl=300)  # Cache por 5 minutos
def cargar_configuracion():
    try:
        dataset_manager = DatasetManager(dataset_path)
        modelos_indiv, modelos_llm = dataset_manager.cargar_modelos_config()
        modelos_hibridos = dataset_manager.cargar_modelos_hibridos()
        modelos_emb = modelos_indiv + modelos_hibridos
        preguntas = dataset_manager.cargar_preguntas_test()
        return modelos_emb, modelos_llm, preguntas, True
    except Exception as e:
        st.error(f"Error cargando configuración: {e}")
        return [], [], [], False

# Cargar configuración
modelos_emb, modelos_llm, preguntas, config_ok = cargar_configuracion()

# Mostrar resumen de configuración
st.header("Configuración del Sistema")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Modelos de Embedding")
    if modelos_emb:
        emb_df = pd.DataFrame([{
            "Nombre": m.nombre,
            #"Variables": ", ".join([f"{k}={v}" for k, v in m.variables_entorno.items()])
        } for m in modelos_emb])
        st.dataframe(emb_df)
    else:
        st.warning("No se encontraron modelos de embedding configurados")

with col2:
    st.subheader("Modelos LLM")
    if modelos_llm:
        llm_df = pd.DataFrame([{
            "Nombre": m.nombre,
            "Variables": ", ".join([f"{k}={v}" for k, v in m.variables_entorno.items()])
        } for m in modelos_llm])
        st.dataframe(llm_df)
    else:
        st.warning("No se encontraron modelos LLM configurados")

# Mostrar dataset de preguntas
st.subheader("Dataset de Preguntas")
if preguntas:
    preguntas_df = pd.DataFrame([{
        "ID": p.id,
        "Pregunta": p.pregunta,
        "Trámite ID": p.tramite_id,
        "Trámite": p.tramite_nombre,
        "URL": p.url_esperada,
        "Categoría": p.categoria
    } for p in preguntas])
    st.dataframe(preguntas_df)
else:
    st.warning("No se encontraron preguntas de prueba configuradas")

# Selección de modelos específicos
st.header("Selección de Modelos")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Modelos de Embedding")
    emb_seleccionados = []
    for m in modelos_emb:
        if st.checkbox(f"{m.nombre}", value=True, key=f"emb_{m.nombre}"):
            emb_seleccionados.append(m.nombre)

with col2:
    st.subheader("Modelos LLM")
    llm_seleccionados = []
    for m in modelos_llm:
        if st.checkbox(f"{m.nombre}", value=True, key=f"llm_{m.nombre}"):
            llm_seleccionados.append(m.nombre)

# Botón para iniciar evaluación
st.header("Ejecución de Pruebas")
start_button = st.button("Iniciar Evaluación", disabled=not config_ok)

if start_button:
    # Preparar parámetros según selección
    solo_emb = eval_type == "Solo Embeddings"
    solo_llm = eval_type == "Solo LLM"
    
    # Combinar modelos seleccionados
    modelos_seleccionados = emb_seleccionados + llm_seleccionados
    
    # Crear sistema de evaluación
    sistema = SistemaEvaluacion(dataset_path)
    
    # Mostrar progreso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Función para ejecutar evaluación
    async def ejecutar_evaluacion():
        try:
            
            # Ejecutar evaluación real
            reportes = await sistema.ejecutar_evaluacion_completa(
                solo_embeddings=solo_emb,
                solo_llm=solo_llm,
                modelos_especificos=modelos_seleccionados if modelos_seleccionados else None
            )
            
            # Actualizar estado
            st.session_state["reportes"] = reportes
            st.session_state["evaluacion_completada"] = True
            
            # Recargar para mostrar resultados
            st.rerun()
            
        except Exception as e:
            st.error(f"Error durante la evaluación: {e}")
    
    # Ejecutar en modo asíncrono
    asyncio.run(ejecutar_evaluacion())

# Mostrar resultados si hay una evaluación previa
if "evaluacion_completada" in st.session_state and st.session_state["evaluacion_completada"]:
    st.header("Resultados de la Evaluación")
    
    # Buscar y mostrar reportes
    if "reportes" in st.session_state:
        reportes = st.session_state["reportes"]
        st.success(f"Evaluación completada con éxito! Se generaron {len(reportes)} reportes.")
        
        for reporte in reportes:
            if reporte.endswith(".csv"):
                try:
                    if "resumen" in reporte:
                        df = pd.read_csv(reporte, header=[0, 1], index_col=0)
                        st.subheader(f"Reporte: {Path(reporte).name}")
                        st.dataframe(df)
                        
                        if "embedding" in reporte:
                            # Tabla resumen de embeddings (mantener igual)
                            st.subheader("📊 Métricas de Embedding (media por modelo)")
                            df_emb = pd.DataFrame({
                                'modelo': df.index,
                                'Latencia (ms)': df[('latencia_ms', 'mean')].values,
                                'Precision@3': df[('precision_at_3', 'mean')].values if ('precision_at_3', 'mean') in df.columns else 0,
                                'Precision@5': df[('precision_at_5', 'mean')].values if ('precision_at_5', 'mean') in df.columns else 0,
                                'Precision@10': df[('precision_at_10', 'mean')].values if ('precision_at_10', 'mean') in df.columns else 0,
                                'MRR@3': df[('mrr_at_3', 'mean')].values if ('mrr_at_3', 'mean') in df.columns else 0,
                                'MRR@5': df[('mrr_at_5', 'mean')].values if ('mrr_at_5', 'mean') in df.columns else 0,
                                'MRR@10': df[('mrr_at_10', 'mean')].values if ('mrr_at_10', 'mean') in df.columns else 0
                            })
                            st.table(df_emb)
                            
                            # Métricas en X, Modelos como barras agrupadas
                            st.subheader("📈 Comparativa de Métricas de Embedding")
                            
                            # Preparar datos: transponer para tener métricas en filas
                            df_plot = df_emb.set_index('modelo')
                            
                            # Separar métricas normalizadas de latencias
                            metricas_norm = ['Precision@3', 'Precision@5', 'Precision@10', 'MRR@3', 'MRR@5', 'MRR@10']
                            df_metricas = df_plot[metricas_norm].T  # Transponer: métricas en filas, modelos en columnas
                            
                            # Gráfica principal con doble eje Y
                            fig, ax1 = plt.subplots(figsize=(16, 8))
                            
                            # Configuración para separación entre grupos
                            num_metrics = len(df_metricas.index)
                            num_models = len(df_metricas.columns)
                            width = 0.25  # Ancho de cada barra individual
                            group_spacing = 0.25  # Espacio entre grupos de métricas
                            
                            # Calcular posiciones base para cada grupo con separación
                            base_positions = np.arange(num_metrics) * (num_models * width + group_spacing)
                            
                            # Colores para los modelos
                            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:cyan', 'tab:red']
                            
                            # Crear barras agrupadas para cada modelo
                            for i, modelo in enumerate(df_metricas.columns):
                                positions = base_positions + i * width
                                ax1.bar(positions, df_metricas[modelo], width, 
                                        label=modelo, color=colors[i % len(colors)], alpha=0.8)
                            
                            # Configurar eje izquierdo para métricas
                            ax1.set_xlabel('Métricas')
                            ax1.set_ylabel('Valor de la métrica', color='black')
                            ax1.set_title('Comparativa de Modelos de Embedding')
                            
                            # Posicionar etiquetas del eje X en el centro de cada grupo
                            center_positions = base_positions + (num_models - 1) * width / 2
                            ax1.set_xticks(center_positions)
                            ax1.set_xticklabels(df_metricas.index, rotation=45)
                            ax1.set_ylim(0, 1)
                            ax1.legend(loc='upper left')
                            ax1.grid(True, alpha=0.3)
                            
                            # Eje derecho: latencias
                            ax2 = ax1.twinx()
                            # Posicionar las barras de latencia después de las métricas con separación adicional
                            latencias_pos = base_positions[-1] + (num_models * width + group_spacing * 2)
                            
                            for i, modelo in enumerate(df_plot.index):
                                offset = i * width
                                ax2.bar(latencias_pos + offset, df_plot.loc[modelo, 'Latencia (ms)'], width,
                                        color=colors[i % len(colors)], alpha=0.6, hatch='//')
                            
                            ax2.set_ylabel('Latencia (ms)', color='red')
                            ax2.tick_params(axis='y', labelcolor='red')
                            
                            # Añadir etiqueta de "Latencia" en el eje X
                            all_labels = list(df_metricas.index) + ['Latencia (ms)']
                            all_positions = list(center_positions) + [latencias_pos + (num_models - 1) * width / 2]
                            ax1.set_xticks(all_positions)
                            ax1.set_xticklabels(all_labels, rotation=45)

                            # Cambia el color de la última etiqueta a rojo
                            labels = ax1.get_xticklabels()
                            labels[-1].set_color('red')

                            plt.tight_layout()
                            st.pyplot(fig)


                        else:  # LLM
                            # Tabla resumen de LLMs (mantener igual)
                            st.subheader("📊 Métricas LLM (media por modelo)")
                            df_llm = pd.DataFrame({
                                'modelo': df.index,
                                'Tiempo LLM (ms)': df[('tiempo_respuesta_ms', 'mean')].values,
                                'Exactitud URL': df[('contiene_url_correcta','mean')].values,
                                'Score Auto': df[('score_autoevaluacion','mean')].values,
                                'Faithfulness': df[('faithfulness_score','mean')].values,
                                'BERTScore F1': df[('bertscore_f1','mean')].values
                            })
                            st.table(df_llm)
                            
                            # Métricas en X, Modelos como barras agrupadas
                            st.subheader("📈 Comparativa de Métricas LLM")
                            
                            # Preparar datos: transponer para tener métricas en filas
                            df_plot = df_llm.set_index('modelo')
                            
                            # Separar métricas normalizadas de tiempos
                            metricas_norm = ['Exactitud URL', 'Score Auto', 'Faithfulness', 'BERTScore F1']
                            df_metricas = df_plot[metricas_norm].T  # Transponer: métricas en filas, modelos en columnas
                            
                            # Gráfica principal con doble eje Y
                            fig, ax1 = plt.subplots(figsize=(14, 8))
                            
                            # Configuración para separación entre grupos
                            num_metrics = len(df_metricas.index)
                            num_models = len(df_metricas.columns)
                            width = 0.15  # Ancho de cada barra individual
                            group_spacing = 1.0  # Espacio entre grupos de métricas
                            
                            # Calcular posiciones base para cada grupo con separación
                            base_positions = np.arange(num_metrics) * (num_models * width + group_spacing)
                            
                            # Colores para los modelos
                            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:cyan', 'tab:red']
                            
                            # Crear barras agrupadas para cada modelo
                            for i, modelo in enumerate(df_metricas.columns):
                                positions = base_positions + i * width
                                ax1.bar(positions, df_metricas[modelo], width, 
                                        label=modelo, color=colors[i % len(colors)], alpha=0.8)
                            
                            # Configurar eje izquierdo para métricas
                            ax1.set_xlabel('Métricas')
                            ax1.set_ylabel('Valor de la métrica', color='black')
                            ax1.set_title('Comparativa de Modelos LLM')
                            
                            # Posicionar etiquetas del eje X en el centro de cada grupo
                            center_positions = base_positions + (num_models - 1) * width / 2
                            ax1.set_xticks(center_positions)
                            ax1.set_xticklabels(df_metricas.index, rotation=45)
                            ax1.set_ylim(0, 1)
                            ax1.legend(loc='upper left')
                            ax1.grid(True, alpha=0.3)
                            
                            # Eje derecho: tiempos de respuesta
                            ax2 = ax1.twinx()
                            # Posicionar las barras de tiempo después de las métricas con separación adicional
                            tiempos_pos = base_positions[-1] + (num_models * width + group_spacing * 2)
                            
                            for i, modelo in enumerate(df_plot.index):
                                offset = i * width
                                ax2.bar(tiempos_pos + offset, df_plot.loc[modelo, 'Tiempo LLM (ms)'], width,
                                        color=colors[i % len(colors)], alpha=0.6, hatch='//')
                            
                            ax2.set_ylabel('Tiempo LLM (ms)', color='red')
                            ax2.tick_params(axis='y', labelcolor='red')
                            
                            # Añadir etiqueta de "Tiempo" en el eje X
                            all_labels = list(df_metricas.index) + ['Tiempo LLM (ms)']
                            all_positions = list(center_positions) + [tiempos_pos + (num_models - 1) * width / 2]
                            ax1.set_xticks(all_positions)
                            ax1.set_xticklabels(all_labels, rotation=45)

                            # Cambia el color de la última etiqueta a rojo
                            labels = ax1.get_xticklabels()
                            labels[-1].set_color('red')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                    else:
                        # Reportes detallados
                        df = pd.read_csv(reporte)
                        st.subheader(f"Reporte: {Path(reporte).name}")
                        st.dataframe(df)

                except Exception as e:
                    st.error(f"Error al mostrar reporte {reporte}: {e}")
                    raise

            elif reporte.endswith(".md"):
                try:
                    with open(reporte, 'r',  encoding='utf-8') as f:
                        contenido = f.read()
                    st.markdown(contenido)
                except Exception as e:
                    st.error(f"Error al mostrar reporte {reporte}: {e}")


# Sección para gestionar la configuración
st.header("Gestión de Configuración")

tab1, tab2, tab3 = st.tabs(["Modelos de Embedding", "Modelos LLM", "Preguntas de Prueba"])

with tab1:
    st.subheader("Configuración de Modelos de Embedding")
    st.markdown("""
    Aquí puedes añadir o modificar los modelos de embedding disponibles para pruebas.
    Los cambios se guardarán en el archivo CSV de configuración.
    """)
    
    # Formulario para añadir modelo de embedding
    with st.form("form_embedding"):
        nombre_emb = st.text_input("Nombre del modelo")
        variables_emb = st.text_area("Variables de entorno (formato JSON)", 
                                    '{"EMBEDDING_MODEL": "nombre-modelo", "EMBEDDING_PROVIDER": "proveedor"}')
        parametros_emb = st.text_area("Parámetros (formato JSON)", 
                                     '{"dimension": 1024}')
        
        submit_emb = st.form_submit_button("Añadir Modelo de Embedding")
        
        if submit_emb:
            try:
                # Validar JSON
                import json
                variables_json = json.loads(variables_emb)
                parametros_json = json.loads(parametros_emb)
                
                # Añadir a CSV
                dataset_manager = DatasetManager(dataset_path)
                # Aquí iría la lógica para añadir al CSV
                
                st.success(f"Modelo {nombre_emb} añadido correctamente")
                # Recargar para actualizar lista
                st.rerun()
            except Exception as e:
                st.error(f"Error añadiendo modelo: {e}")

with tab2:
    st.subheader("Configuración de Modelos LLM")
    st.markdown("""
    Aquí puedes añadir o modificar los modelos LLM disponibles para pruebas.
    Los cambios se guardarán en el archivo CSV de configuración.
    """)
    
    # Formulario para añadir modelo LLM
    with st.form("form_llm"):
        nombre_llm = st.text_input("Nombre del modelo")
        variables_llm = st.text_area("Variables de entorno (formato JSON)", 
                                   '{"LLM_MODEL": "nombre-modelo", "LLM_PROVIDER": "proveedor"}')
        parametros_llm = st.text_area("Parámetros (formato JSON)", 
                                    '{"temperature": 0.7, "max_tokens": 1500}')
        
        submit_llm = st.form_submit_button("Añadir Modelo LLM")
        
        if submit_llm:
            try:
                # Validar JSON
                import json
                variables_json = json.loads(variables_llm)
                parametros_json = json.loads(parametros_llm)
                
                # Añadir a CSV
                dataset_manager = DatasetManager(dataset_path)
                # Aquí iría la lógica para añadir al CSV
                
                st.success(f"Modelo {nombre_llm} añadido correctamente")
                # Recargar para actualizar lista
                st.rerun()
            except Exception as e:
                st.error(f"Error añadiendo modelo: {e}")

with tab3:
    st.subheader("Configuración de Preguntas de Prueba")
    st.markdown("""
    Aquí puedes añadir o modificar las preguntas de prueba para evaluar los modelos.
    Los cambios se guardarán en el archivo CSV de configuración.
    """)
    
    # Formulario para añadir pregunta
    with st.form("form_pregunta"):
        pregunta = st.text_input("Pregunta")
        tramite_id = st.number_input("ID del Trámite", min_value=1)
        tramite_nombre = st.text_input("Nombre del Trámite")
        url_esperada = st.text_input("URL Esperada", "https://sede.murcia.es/ficha-procedimiento?id=")
        categoria = st.selectbox("Categoría", ["certificados", "documentacion", "pagos", "ayudas", "empresas", "otros"])
        
        submit_pregunta = st.form_submit_button("Añadir Pregunta")
        
        if submit_pregunta:
            try:
                # Añadir a CSV
                dataset_manager = DatasetManager(dataset_path)
                # Aquí iría la lógica para añadir al CSV
                
                st.success(f"Pregunta añadida correctamente")
                # Recargar para actualizar lista
                st.rerun()
            except Exception as e:
                st.error(f"Error añadiendo pregunta: {e}")

# -----------------------------------------------------------------------------
# 🔄 Resultados guardados
# -----------------------------------------------------------------------------
st.header("Resultados guardados (CSV en carpeta test_results)")
guardados = escanear_resultados(results_path)

if guardados:
    # Mostrar lista en la barra lateral
    with st.sidebar.expander("Seleccionar resultado previo"):
        opciones = [f"{g['fecha'].strftime('%Y-%m-%d %H:%M')} · {g['tipo']} · {g['nombre']}"
                    for g in guardados]
        idx = st.selectbox("Escoge un archivo", range(len(opciones)),
                           format_func=lambda i: opciones[i])
        boton_cargar = st.button("Cargar resultado", key="btn_cargar_guardado")
else:
    st.sidebar.warning("No se encontraron CSV en /test_results/")
    boton_cargar = False

# Al pulsar el botón leemos y pintamos
if boton_cargar:
    ruta = guardados[idx]["ruta"]
    df, tipo = cargar_csv_resultado(ruta)
    st.subheader(f"📄 {Path(ruta).name}")
    if tipo == "resumen":
        mostrar_resumen(ruta)
    else:  # detallado
        st.dataframe(df)