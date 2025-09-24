# 🧬 Algoritmo Genético para la Selección de Agentes en Catán

Este proyecto implementa un **algoritmo genético (AG)** para optimizar la selección de agentes en partidas simuladas de *Catán*.  
El objetivo principal es utilizar **métodos evolutivos** para mejorar las estrategias de los agentes y evaluar su rendimiento mediante experimentación de hiperparámetros, incluyendo ejecuciones en entornos cloud como **AWS SageMaker**.

---

## 🎯 Objetivo del Proyecto

- Desarrollar e implementar un **algoritmo genético en Python** para optimizar la selección de agentes.  
- Representar soluciones como **vectores de probabilidad**, donde cada valor indica la probabilidad de elegir un agente.  
- Evaluar el *fitness* en base al desempeño en múltiples partidas simuladas.  
- Experimentar con distintas configuraciones de hiperparámetros (tamaño de población, probabilidad de cruce/mutación, número de generaciones, etc.).  
- Ejecutar diversos de los experimentos en **Amazon SageMaker** para pruebas a gran escala.  

---

## ⚙️ Descripción de la Implementación

### Componentes del Algoritmo Genético
- **Codificación de la solución**: vector de probabilidades de selección de agentes.  
- **Inicialización de la población**: aleatoria o basada en heurísticas.  
- **Operadores evolutivos**:  
  - Cruce  
  - Mutación  
  - Selección (torneo, ruleta, elitismo)  
  - Estrategias de reemplazo  
- **Evaluación del fitness**:  
  - Partidas simuladas con [PyCatan](https://github.com/jaumejordan/PyCatan) y los agentes proporcionados.  
  - Métricas basadas en puntos de victoria y resultados de la partida.  
  - Consideración del orden de juego y robustez estadística (N partidas por evaluación).   

---

## 📂 Estructura del Repositorio

- `genetic.py` → Implementación principal del algoritmo genético y pipeline de experimentos.  
- `results_csv/` → Registros en CSV de los experimentos (fitness medio, máximo y mejor individuo por generación).  
- `initial_population/` → Poblaciones iniciales utilizadas en los experimentos.  
- `results.ipynb` → Notebook con el análisis de resultados, hiperparámetros y reflexiones.  
- `Informe_AWS_Ejecución_Algoritmo_Genético.pdf` → Memoria con la ejecución en **AWS SageMaker** usando `launcher.py`, configuración, errores encontrados y soluciones aplicadas.  

