# Detección predictiva de fraude en transacciones electrónicas

## Proyecto final: Machine Learning y simulación de reglas de riesgo

Este proyecto continúa el trabajo realizado en el [Proyecto Fraude Midterm](https://github.com/wilydus-png/Proyecto-Fraude-Midterm), ampliando el análisis exploratorio y las reglas de detección mediante modelos supervisados de *Machine Learning*, simulaciones de umbrales de alerta, comparaciones de capacidad operativa y análisis económico hipotético.

**Objetivo:** comparar la detección y priorización de transacciones potencialmente fraudulentas mediante tres reglas tradicionales y tres modelos predictivos, considerando la detección de fraude, las falsas alertas y posibles costos operativos.

> **Alcance:** estudio académico retrospectivo. No es un sistema de autorización de pagos en producción ni debe utilizarse para bloquear transacciones reales sin validaciones adicionales.

## 1. Datos utilizados

Se emplea el conjunto de transacciones sintéticas de tarjetas conocido como *Credit Card Transactions Fraud Detection*, distribuido en dos archivos:

- `fraudTrain.csv`: **1.296.675** transacciones.
- `fraudTest.csv`: **555.719** transacciones.
- **Total:** 1.852.394 transacciones; **9.651** etiquetadas como fraude.
- Variable objetivo: `is_fraud` (`0` = legítima, `1` = fraudulenta).

Se respetó el conjunto original de prueba y se dividió cronológicamente el archivo de entrenamiento:

| Conjunto | Transacciones | Fraudes | Tasa de fraude |
|---|---:|---:|---:|
| Entrenamiento | 1.037.340 | 5.968 | 0,575 % |
| Validación | 259.335 | 1.538 | 0,593 % |
| Prueba final | 555.719 | 2.145 | 0,386 % |

La separación temporal permitió entrenar con operaciones anteriores y evaluar el comportamiento en periodos posteriores. No se utilizaron las etiquetas de prueba para entrenar los modelos ni para seleccionar sus umbrales.

## 2. Herramientas

- Python 3
- Jupyter Notebook
- pandas, NumPy, Matplotlib, Seaborn
- scikit-learn

## 3. Metodología

1. **Preparación:** lectura de los archivos CSV, revisión de tipos de datos y valores faltantes, y análisis del desbalance de clases.
2. **Ingeniería de características:** extracción de hora, día de la semana y fin de semana, y cálculo de `distancia_km` mediante Haversine.
3. **Selección inicial de predictores:** `amt`, `hora`, `dia_semana`, `fin_semana`, `distancia_km`, `city_pop`, `category` y `merchant`.
4. **Preprocesamiento:** estandarización de variables numéricas con `StandardScaler` y codificación categórica mediante `OneHotEncoder`, integradas con `ColumnTransformer` y `Pipeline`. Los transformadores se ajustaron solamente con entrenamiento.
5. **Entrenamiento:** Regresión Logística, Árbol de Decisión y Random Forest, con ponderaciones para atender el desbalance.
6. **Validación:** Precision, Recall, F1-Score, Average Precision, matrices de confusión, comparación de umbrales y simulación Top-K para capacidades de revisión fijas.
7. **Reglas tradicionales:** implementación de R1, R2 y R3 para su comparación retrospectiva.
8. **Evaluación final:** aplicación de configuraciones fijadas previamente al conjunto independiente `fraudTest.csv`.
9. **Simulaciones operativas y económicas:** eficiencia por alerta, alternativa híbrida R3 + Random Forest, costo hipotético y análisis de sensibilidad.

### Reglas evaluadas

| Regla | Condición utilizada en este Notebook |
|---|---|
| **R1** | `amt >= 200` |
| **R2** | R1 y hora entre 22:00 y 03:59 |
| **R3** | R2 y categoría en `shopping_net`, `misc_net` o `grocery_pos` |

**Nota de trazabilidad:** antes de dar por definitiva la continuidad con el Midterm, se debe cotejar la definición exacta de estas tres reglas con el código original, incluidos operadores, horarios y categorías.

### Modelos y umbrales de la evaluación final

| Modelo | Configuración final |
|---|---|
| Regresión Logística | Umbral 0,80 |
| Árbol de Decisión | Umbral 0,90 |
| Random Forest | Umbral 0,70 |

Estos umbrales se seleccionaron exploratoriamente durante la validación. Las puntuaciones de los modelos ponderados no fueron calibradas como probabilidades reales de fraude.

## 4. Resultados de la evaluación final

Los resultados corresponden exclusivamente a las **555.719 transacciones del conjunto de prueba**, con **2.145 fraudes reales**.

| Estrategia | Alertas | Fraudes detectados | Fraudes no detectados | Falsas alertas | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 | 25.353 | 1.611 | 534 | 23.742 | 6,35 % | 75,10 % | 11,72 % |
| R2 | 7.035 | 1.363 | 782 | 5.672 | 19,37 % | 63,54 % | 29,69 % |
| R3 | 3.898 | 1.069 | 1.076 | 2.829 | 27,42 % | 49,84 % | 35,38 % |
| Regresión Logística | 8.252 | 1.395 | 750 | 6.857 | 16,90 % | 65,03 % | 26,83 % |
| **Árbol de Decisión** | 13.254 | **2.001** | 144 | 11.253 | 15,10 % | **93,29 %** | 25,99 % |
| **Random Forest** | **2.077** | 1.275 | 870 | **802** | **61,39 %** | 59,44 % | **60,40 %** |

**Interpretación:** el Árbol de Decisión obtuvo la mayor cobertura de fraude (*Recall*), mientras que Random Forest logró la mayor precisión y el mayor F1-Score bajo las configuraciones seleccionadas. Ninguna estrategia domina en todas las métricas ni utiliza la misma cantidad de alertas.

### Priorización Top-K sobre validación

Con una capacidad fija de revisión, los modelos se compararon ordenando las transacciones de mayor a menor puntuación de riesgo:

| Capacidad de revisión | Mejor modelo | Fraudes detectados | Recall@K |
|---|---|---:|---:|
| 1.000 | Random Forest | 820 | 53,32 % |
| 3.000 | Random Forest | 1.134 | 73,73 % |
| 5.000 | Árbol de Decisión | 1.294 | 84,14 % |
| 10.000 | Árbol de Decisión | 1.494 | 97,14 % |

Los volúmenes corresponden a todo el periodo de validación, **no a un límite diario**. Las puntuaciones empatadas pueden afectar el orden Top-K, especialmente en los árboles.

### Estrategia híbrida

Se probó una priorización que colocaba primero las **2.069 transacciones** activadas por R3 y completaba las **5.000 alertas** con Random Forest. Detectó **1.158 fraudes**, frente a **1.162** de Random Forest solo y **1.294** del Árbol de Decisión con la misma capacidad. La combinación ensayada **no mejoró** el resultado frente a Random Forest independiente.

## 5. Análisis económico hipotético

Se calculó el costo ilustrativo:

`Costo total = (número de alertas × costo de revisión) + (fraudes no detectados × pérdida media supuesta)`

Con **USD 2 por revisión** y **USD 100 por fraude no detectado**, el menor costo estimado correspondió al **Árbol de Decisión: USD 40.908**.

En una sensibilidad de **25 combinaciones** de costos de revisión (USD 1, 2, 5, 10 y 20) y pérdidas por fraude (USD 25, 50, 100, 250 y 500), el Árbol de Decisión presentó el menor costo en **15 escenarios** y Random Forest en **10**.

Estos importes son **supuestos académicos**, no pérdidas, ahorros o rentabilidad efectivamente observados. El cálculo supone que todas las alertas se revisan y que los fraudes detectados pueden tratarse oportunamente.

## 6. Cómo ejecutar el proyecto

1. Descargar o clonar este repositorio.
2. Disponer localmente de `fraudTrain.csv` y `fraudTest.csv` del conjunto de datos original. **Los CSV no se incluyen necesariamente en GitHub** por su tamaño y deben obtenerse de su fuente autorizada.
3. Colocar los CSV en la misma carpeta desde la que se ejecuta Jupyter, o adaptar las rutas de lectura del Notebook.
4. Instalar las dependencias si hace falta:

   ```bash
   python -m pip install pandas numpy matplotlib seaborn scikit-learn notebook
   ```

5. Abrir Jupyter:

   ```bash
   jupyter notebook
   ```

6. Abrir `Proyecto_Fraude_Modelamiento.ipynb` **(ajustar este nombre si el archivo publicado tiene otro)**.
7. Ejecutar las celdas **en orden**, desde el inicio, hasta generar los resultados y gráficos. El entrenamiento puede tardar según los recursos del equipo.

La ruta del código toma como referencia la carpeta de trabajo actual (`Path.cwd()`), por lo que la ubicación de los CSV debe concordar con ella.

## 7. Limitaciones

- Los datos son históricos y sintéticos; los resultados no demuestran eficacia en una entidad financiera real.
- Las reglas proceden de un análisis exploratorio previo que consideró datos posteriormente destinados a prueba, por lo que su evaluación **no es totalmente independiente del proceso original de diseño**.
- La selección de características y de modelos fue exploratoria; la evaluación temporal ayuda a controlar la fuga de información, pero no sustituye una validación prospectiva.
- La información de coordenadas no necesariamente refleja la ubicación física del titular durante la compra.
- Las probabilidades estimadas por clasificadores con ponderación de clases no fueron calibradas para representar riesgos absolutos.
- Las estrategias comparadas pueden generar números de alertas distintos; por ello también se utilizaron simulaciones Top-K.
- La simulación no reproduce latencia de autorización, procesos de bloqueo, revisión humana, recuperaciones, consecuencias para clientes ni disponibilidad real de datos en línea.
- El modelo económico usa importes hipotéticos y una estructura de costos simplificada.
- Antes de una aplicación real se necesitarían monitoreo de cambios en datos y desempeño, validaciones de seguridad, equidad, gobernanza y controles operativos.

## 8. Conclusión

El proyecto muestra que las reglas tradicionales y el aprendizaje automático pueden evaluarse bajo criterios comparables, pero que **no existe una alternativa universalmente mejor**: el Árbol de Decisión destacó por detectar la mayor proporción de fraudes, mientras que Random Forest destacó por su precisión y la priorización de alertas. La selección de una estrategia requiere considerar, además de las métricas predictivas, la capacidad operativa y los costos de error.

## 9. Antecedente académico

- [Proyecto Fraude Midterm — repositorio original](https://github.com/wilydus-png/Proyecto-Fraude-Midterm)

