# Análisis de patrones transaccionales para la detección de fraude

## Proyecto Midterm – Ciencia de Datos

### Descripción

Este proyecto analiza patrones presentes en transacciones con tarjetas de crédito
con el propósito de diseñar y evaluar reglas exploratorias para la detección de
operaciones potencialmente fraudulentas.

El proceso incluye integración, limpieza, transformación, análisis exploratorio,
visualización, almacenamiento en SQLite y evaluación de reglas mediante métricas
de clasificación.

## Fuente de datos

Se utiliza el **Credit Card Transactions Fraud Detection Dataset**, publicado
en Kaggle por Kartik Shenoy.

Dataset:
https://www.kaggle.com/datasets/kartik2112/fraud-detection

Los archivos originales utilizados son:

- `fraudTrain.csv`
- `fraudTest.csv`

Debido a su tamaño, los archivos de datos no se incluyen en este repositorio
y deben descargarse desde la fuente original.

## Dimensiones del conjunto integrado

- Transacciones analizadas: 1,852,394
- Transacciones legítimas: 1,842,743
- Transacciones fraudulentas: 9,651
- Tasa global de fraude: 0.521 %
- Período analizado: 2019–2020

## Metodología

El proyecto comprende las siguientes etapas:

1. Carga de los archivos originales.
2. Integración de los conjuntos de entrenamiento y prueba.
3. Verificación de valores nulos y registros duplicados.
4. Eliminación de variables de índice no necesarias.
5. Conversión de variables de fecha.
6. Generación de variables temporales.
7. Análisis exploratorio de patrones de fraude.
8. Diseño de reglas exploratorias de detección.
9. Evaluación mediante precisión, recall, F1-Score y tasa de falsos positivos.
10. Generación de visualizaciones.
11. Almacenamiento y consulta de datos mediante SQLite.

## Reglas exploratorias evaluadas

- **R1:** monto >= USD 200.
- **R2:** monto >= USD 200 y horario entre 22:00 y 03:59.
- **R3:** monto >= USD 200, horario entre 22:00 y 03:59 y categorías seleccionadas según la mayor tasa de fraude observada.

Las reglas muestran diferentes relaciones entre capacidad de detección y
generación de falsos positivos, por lo que su evaluación considera varias
métricas y no únicamente la cantidad de fraudes identificados.

## Principales resultados

| Regla | Precisión | Recall | F1-Score |
|---|---:|---:|---:|
| R1 | 8.38 % | 75.80 % | 15.09 % |
| R2 | 24.70 % | 64.29 % | 35.69 % |
| R3 | 33.38 % | 49.77 % | 39.96 % |

Los resultados evidencian un intercambio entre cobertura del fraude y
generación de falsos positivos. La incorporación progresiva de condiciones
aumenta la precisión de las alertas, pero reduce el porcentaje total de fraudes
detectados.

## Archivos principales

- `Proyecto_Fraude_Midterm.ipynb`: desarrollo completo del análisis.
- `descripcion_dataset.txt`: descripción y metadata de las variables.
- `distribucion_fraude.png`: distribución de transacciones.
- `tasa_fraude_por_hora.png`: comportamiento temporal del fraude.
- `tasa_fraude_por_categoria.png`: tasa de fraude por categoría comercial.
- `comparacion_reglas_precision_recall.png`: comparación de las reglas.

## Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- SQLite
- JupyterLab

## Nota

Las reglas desarrolladas tienen un propósito académico y exploratorio. Los
resultados corresponden exclusivamente al conjunto de datos analizado y no
deben interpretarse como reglas universales para sistemas reales de prevención
de fraude.
