# ejemplo_uso.py
"""
Ejemplo Simple de Uso del Sistema de Pruebas Automatizadas
Sede Electrónica del Ayuntamiento de Murcia

Este script muestra cómo usar el sistema de pruebas paso a paso.
"""

import asyncio
import os
from pathlib import Path

# Importar el sistema de pruebas
from backend.test_automatizado import (
    DatasetManager, ModelConfig, PreguntaTest,
    EvaluadorEmbeddings, EvaluadorLLM, ModelSwitcher
)

async def ejemplo_basico():
    """Ejemplo básico de evaluación de un solo modelo"""
    print("🚀 Ejemplo básico de evaluación de modelos")
    
    # 1. Configurar un modelo de embedding de ejemplo
    config_embedding = ModelConfig(
        nombre="mxbai-embed-large",
        tipo="embedding",
        parametros={"dimension": 1024},
        variables_entorno={
            "EMBEDDING_MODEL": "mxbai-embed-large",
            "EMBEDDING_PROVIDER": "ollama"
        }
    )
    
    # 2. Configurar un modelo LLM de ejemplo
    config_llm = ModelConfig(
        nombre="gemma2:27b",
        tipo="llm",
        parametros={"temperature": 0.7, "max_tokens": 1500},
        variables_entorno={
            "LLM_MODEL": "gemma2:27b",
            "LLM_PROVIDER": "ollama"
        }
    )
    
    # 3. Crear algunas preguntas de prueba
    preguntas_test = [
        PreguntaTest(
            id=1,
            pregunta="¿Cómo puedo solicitar el certificado de empadronamiento?",
            tramite_id=1001,
            tramite_nombre="Certificado de Empadronamiento",
            url_esperada="https://sede.murcia.es/ficha-procedimiento?id=1001",
            categoria="certificados"
        ),
        PreguntaTest(
            id=2,
            pregunta="¿Qué documentos necesito para renovar el DNI?",
            tramite_id=2001,
            tramite_nombre="Renovación DNI",
            url_esperada="https://sede.murcia.es/ficha-procedimiento?id=2001",
            categoria="documentacion"
        )
    ]
    
    # 4. Crear evaluadores
    evaluador_emb = EvaluadorEmbeddings(k=10)
    evaluador_llm = EvaluadorLLM()
    switcher = ModelSwitcher()
    
    try:
        # 5. Evaluar modelo de embedding
        print("\n🔍 Evaluando modelo de embedding...")
        resultados_emb = await evaluador_emb.evaluar_modelo(
            config_embedding, preguntas_test, switcher
        )
        
        # Mostrar resultados de embedding
        print("Resultados de Embedding:")
        for resultado in resultados_emb:
            print(f"  Pregunta {resultado.pregunta_id}:")
            print(f"    Precisión@10: {resultado.precision_at_k:.3f}")
            print(f"    MRR: {resultado.mrr:.3f}")
            print(f"    Latencia: {resultado.latencia_ms:.1f}ms")
            print(f"    Posición correcta: {resultado.posicion_correcto}")
        
        # 6. Evaluar modelo LLM
        print("\n🤖 Evaluando modelo LLM...")
        resultados_llm = await evaluador_llm.evaluar_modelo(
            config_llm, preguntas_test, switcher
        )
        
        # Mostrar resultados de LLM
        print("Resultados de LLM:")
        for resultado in resultados_llm:
            print(f"  Pregunta {resultado.pregunta_id}:")
            print(f"    URL correcta: {resultado.contiene_url_correcta}")
            print(f"    Score auto: {resultado.score_autoevaluacion:.1f}")
            print(f"    Tiempo: {resultado.tiempo_respuesta_ms:.1f}ms")
        
        print("\n✅ Evaluación completada exitosamente!")
        
    except Exception as e:
        print(f"❌ Error durante la evaluación: {e}")
    finally:
        # Restaurar entorno original
        switcher.restaurar_entorno()

async def ejemplo_comparacion_modelos():
    """Ejemplo de comparación entre múltiples modelos"""
    print("\n🔄 Ejemplo de comparación entre modelos")
    
    # Configurar múltiples modelos de embedding
    modelos_embedding = [
        ModelConfig(
            nombre="mxbai-embed-large",
            tipo="embedding",
            parametros={"dimension": 1024},
            variables_entorno={
                "EMBEDDING_MODEL": "mxbai-embed-large",
                "EMBEDDING_PROVIDER": "ollama"
            }
        ),
        ModelConfig(
            nombre="nomic-embed-text",
            tipo="embedding",
            parametros={"dimension": 768},
            variables_entorno={
                "EMBEDDING_MODEL": "nomic-embed-text", 
                "EMBEDDING_PROVIDER": "ollama"
            }
        )
    ]
    
    # Una pregunta de prueba simple
    pregunta_test = [
        PreguntaTest(
            id=1,
            pregunta="¿Cómo solicito ayudas sociales?",
            tramite_id=4001,
            tramite_nombre="Solicitud de Ayudas Sociales",
            url_esperada="https://sede.murcia.es/ficha-procedimiento?id=4001",
            categoria="ayudas"
        )
    ]
    
    evaluador = EvaluadorEmbeddings(k=5)
    switcher = ModelSwitcher()
    
    resultados_comparacion = {}
    
    try:
        for modelo in modelos_embedding:
            print(f"  📝 Evaluando {modelo.nombre}...")
            resultados = await evaluador.evaluar_modelo(modelo, pregunta_test, switcher)
            
            if resultados:
                resultado = resultados[0]  # Solo una pregunta
                resultados_comparacion[modelo.nombre] = {
                    'precision': resultado.precision_at_k,
                    'mrr': resultado.mrr,
                    'latencia': resultado.latencia_ms
                }
        
        # Mostrar comparación
        print("\n📊 Comparación de modelos:")
        print("Modelo".ljust(20) + "Precisión".ljust(12) + "MRR".ljust(8) + "Latencia")
        print("-" * 50)
        
        for modelo, metricas in resultados_comparacion.items():
            print(f"{modelo.ljust(20)}{metricas['precision']:.3f}".ljust(32) + 
                  f"{metricas['mrr']:.3f}".ljust(8) + 
                  f"{metricas['latencia']:.1f}ms")
        
        # Determinar el mejor modelo
        mejor_modelo = max(resultados_comparacion.items(), 
                          key=lambda x: x[1]['precision'])
        print(f"\n🏆 Mejor modelo: {mejor_modelo[0]} " +
              f"(Precisión: {mejor_modelo[1]['precision']:.3f})")
        
    except Exception as e:
        print(f"❌ Error en comparación: {e}")
    finally:
        switcher.restaurar_entorno()

def ejemplo_gestion_dataset():
    """Ejemplo de gestión del dataset de pruebas"""
    print("\n📁 Ejemplo de gestión del dataset")
    
    # Crear gestor de dataset
    dataset_manager = DatasetManager("./test_data_ejemplo")
    
    try:
        # Cargar configuraciones (se crean automáticamente si no existen)
        modelos_emb, modelos_llm = dataset_manager.cargar_modelos_config()
        preguntas = dataset_manager.cargar_preguntas_test()
        
        print(f"📊 Modelos de embedding cargados: {len(modelos_emb)}")
        for modelo in modelos_emb:
            print(f"  - {modelo.nombre}")
        
        print(f"📊 Modelos LLM cargados: {len(modelos_llm)}")
        for modelo in modelos_llm:
            print(f"  - {modelo.nombre}")
        
        print(f"📊 Preguntas de prueba cargadas: {len(preguntas)}")
        for pregunta in preguntas:
            print(f"  - P{pregunta.id}: {pregunta.pregunta[:50]}...")
        
        print("\n✅ Dataset cargado correctamente!")
        print(f"📁 Archivos CSV creados en: ./test_data_ejemplo/")
        
    except Exception as e:
        print(f"❌ Error gestionando dataset: {e}")

async def main():
    """Función principal con todos los ejemplos"""
    print("🎯 Ejemplos de Uso del Sistema de Pruebas Automatizadas")
    print("=" * 60)
    
    # Ejemplo 1: Dataset
    ejemplo_gestion_dataset()
    
    # Ejemplo 2: Evaluación básica
    await ejemplo_basico()
    
    # Ejemplo 3: Comparación
    await ejemplo_comparacion_modelos()
    
    print("\n🎉 Todos los ejemplos completados!")
    print("\nPróximos pasos:")
    print("1. Personaliza los archivos CSV en ./test_data/")
    print("2. Ejecuta: python ejecutor_pruebas.py")
    print("3. O usa la interfaz: streamlit run interfaz_pruebas.py")

if __name__ == "__main__":
    asyncio.run(main())