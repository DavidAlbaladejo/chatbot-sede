# ejecutor_pruebas.py
"""
Ejecutor Principal del Sistema de Pruebas Automatizadas
Sede Electrónica del Ayuntamiento de Murcia
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import List, Optional
import logging
import pandas as pd


from backend.test_automatizado import (
    DatasetManager, ModelSwitcher, EvaluadorEmbeddings, EvaluadorLLM, 
    GeneradorReportes, ModelConfig, PreguntaTest
)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SistemaEvaluacion:
    """Sistema principal de evaluación de modelos"""
    
    def __init__(self, dataset_path: str = "./test_data"):
        self.dataset_manager = DatasetManager(dataset_path)
        self.model_switcher = ModelSwitcher()
        self.evaluador_emb = EvaluadorEmbeddings(ks=[3, 5, 10])
        self.evaluador_llm = EvaluadorLLM()
        self.generador_reportes = GeneradorReportes()
        
    async def ejecutar_evaluacion_completa(self, 
                                         solo_embeddings: bool = False,
                                         solo_llm: bool = False,
                                         modelos_especificos: Optional[List[str]] = None):
        """Ejecuta evaluación completa del sistema"""
        logger.info("🚀 Iniciando evaluación completa del sistema RAG")
        
        # Cargar configuraciones
        modelos_indiv, modelos_llm = self.dataset_manager.cargar_modelos_config()
        modelos_hibridos = self.dataset_manager.cargar_modelos_hibridos()
        modelos_emb = modelos_indiv + modelos_hibridos
        preguntas = self.dataset_manager.cargar_preguntas_test()

        logger.info(f"📊 Cargados {len(modelos_emb)} modelos de embedding")
        logger.info(f"📊 Cargados {len(modelos_llm)} modelos LLM")
        logger.info(f"📊 Cargadas {len(preguntas)} preguntas de prueba")

        # Filtrar modelos si se especifican
        if modelos_especificos:
            modelos_emb = [m for m in modelos_emb if m.nombre in modelos_especificos]
            modelos_llm = [m for m in modelos_llm if m.nombre in modelos_especificos]

        # Construir el diccionario de configs individuales por nombre
        configs_dict = {m.nombre: m for m in modelos_indiv}
        logger.info(f"📋 Configuraciones individuales cargadas: {list(configs_dict.keys())}")

        resultados_emb = []
        resultados_llm = []

        try:
            # Evaluar modelos LLM
            if not solo_embeddings:
                logger.info("🤖 Evaluando modelos LLM...")
                for modelo in modelos_llm:
                    try:
                        logger.info(f"  📝 Evaluando {modelo.nombre}...")
                        resultados_modelo = await self.evaluador_llm.evaluar_modelo(
                            modelo, preguntas, self.model_switcher
                        )
                        resultados_llm.extend(resultados_modelo)
                        logger.info(f"  ✅ {modelo.nombre} completado")
                    except Exception as e:
                        logger.error(f"  ❌ Error evaluando {modelo.nombre}: {e}")

            # Evaluar modelos de embedding (individuales e híbridos)
            if not solo_llm:
                logger.info("🔍 Evaluando modelos de embedding...")
                for modelo in modelos_emb:
                    try:
                        logger.info(f"  📝 Evaluando {modelo.nombre}...")
                        # PASAR configs_dict SOLO para híbridos
                        if hasattr(modelo, "modelos_componentes"):  # Es híbrido
                            logger.info(f"  🏗️ Modelo híbrido detectado: {modelo.nombre}")
                            logger.info(f"  🏗️ Modelos componentes: {modelo.modelos_componentes}")
                            logger.info(f"  🏗️ Config individuales: {configs_dict}")
                            resultados_modelo = await self.evaluador_emb.evaluar_modelo(
                                modelo, preguntas, self.model_switcher, configs_dict
                            )
                        else:  # Es individual
                            resultados_modelo = await self.evaluador_emb.evaluar_modelo(
                                modelo, preguntas, self.model_switcher
                            )
                        resultados_emb.extend(resultados_modelo)
                        logger.info(f"  ✅ {modelo.nombre} completado")
                    except Exception as e:
                        logger.error(f"  ❌ Error evaluando {modelo.nombre}: {e}")
                        
            # Generar reportes
            logger.info("📋 Generando reportes...")
            reportes_generados = []
            
            if resultados_emb:
                reporte_emb = self.generador_reportes.generar_reporte_embeddings(resultados_emb)
                reportes_generados.append(reporte_emb)
                logger.info(f"  📄 Reporte embeddings: {reporte_emb}")
            
            if resultados_llm:
                reporte_llm = self.generador_reportes.generar_reporte_llm(resultados_llm)
                reportes_generados.append(reporte_llm)
                logger.info(f"  📄 Reporte LLM: {reporte_llm}")
            
            if resultados_emb and resultados_llm:
                logger.info(f"  📄 Generando reporte comparativo")
                reporte_comp = self.generador_reportes.generar_reporte_comparativo(
                    resultados_emb, resultados_llm
                )
                reportes_generados.append(reporte_comp)
                logger.info(f"  📄 Reporte comparativo: {reporte_comp}")
            
            # Mostrar resumen
            self._mostrar_resumen(resultados_emb, resultados_llm)
            
            return reportes_generados
            
        finally:
            # Restaurar entorno original
            self.model_switcher.restaurar_entorno()
            
    def _mostrar_resumen(self, resultados_emb: list, resultados_llm: list):
        """Muestra resumen de resultados en consola"""
        print("\n" + "="*60)
        print("📊 RESUMEN DE EVALUACIÓN")
        print("="*60)

        if resultados_emb:
            print("\n🔍 MODELOS DE EMBEDDING:")

            # Extraer todas las métricas posibles de los resultados
            rows = []
            for r in resultados_emb:
                row = {'modelo': r.modelo, 'latencia_ms': getattr(r, 'latencia_ms', None)}
                # Si las métricas están en un dict (nuevo formato)
                if hasattr(r, 'metricas'):
                    row.update(r.metricas)
                # Si están como atributos (antiguo formato, por compatibilidad)
                else:
                    for attr in ['precision_at_k', 'mrr']:
                        if hasattr(r, attr):
                            row[attr] = getattr(r, attr)
                rows.append(row)

            df_emb = pd.DataFrame(rows)
            # Agrupa por modelo y calcula media de todas las columnas numéricas excepto 'modelo'
            summary_emb = df_emb.groupby('modelo').mean(numeric_only=True).round(3)

            for modelo, row in summary_emb.iterrows():
                print(f"  {modelo}:")
                # Busca todas las métricas que sean precisión o mrr para mostrar
                for col in row.index:
                    if col.startswith('precision_at_'):
                        print(f"    Precisión@{col.split('_')[-1]}: {row[col]:.3f}")
                    if col.startswith('mrr_at_'):
                        print(f"    MRR@{col.split('_')[-1]}: {row[col]:.3f}")
                if 'latencia_ms' in row:
                    print(f"    Latencia media: {row['latencia_ms']:.1f}ms")

        if resultados_llm:
            print("\n🤖 MODELOS LLM:")
            
            df_llm = pd.DataFrame([{
                'modelo': r.modelo,
                'url_correcta': r.contiene_url_correcta,
                'score_auto': r.score_autoevaluacion,
                'faithfulness_score': getattr(r, 'faithfulness_score', 0.0),
                'bertscore_f1': getattr(r, 'bertscore_f1', 0.0),
                'tiempo': r.tiempo_respuesta_ms
            } for r in resultados_llm])

            summary_llm = df_llm.groupby('modelo').agg({
                'url_correcta': 'mean',
                'score_auto': 'mean',
                'faithfulness_score': 'mean',
                'bertscore_f1': 'mean',
                'tiempo': 'mean'
            }).round(3)

            for modelo, row in summary_llm.iterrows():
                print(f"  {modelo}:")
                print(f"    Exactitud URL:  {row['url_correcta']:.3f}")
                print(f"    Score Auto:     {row['score_auto']:.1f}")
                print(f"    Faithfulness:   {row['faithfulness_score']:.3f}")
                print(f"    BERTScore F1:   {row['bertscore_f1']:.3f}")
                print(f"    Tiempo:         {row['tiempo']:.1f}ms")

        print("\n" + "="*60)

async def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Sistema de Evaluación de Modelos RAG')
    parser.add_argument('--dataset-path', default='./test_data', 
                       help='Ruta al directorio de datos de prueba')
    parser.add_argument('--solo-embeddings', action='store_true',
                       help='Evaluar solo modelos de embedding')
    parser.add_argument('--solo-llm', action='store_true',
                       help='Evaluar solo modelos LLM')
    parser.add_argument('--modelos', nargs='+',
                       help='Modelos específicos a evaluar')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Modo verbose')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Crear sistema de evaluación
    sistema = SistemaEvaluacion(args.dataset_path)
    
    try:
        reportes = await sistema.ejecutar_evaluacion_completa(
            solo_embeddings=args.solo_embeddings,
            solo_llm=args.solo_llm,
            modelos_especificos=args.modelos
        )
        
        print(f"\n✅ Evaluación completada exitosamente!")
        print(f"📁 Reportes generados: {len(reportes)}")
        for reporte in reportes:
            print(f"   - {reporte}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Evaluación interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error durante la evaluación: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())