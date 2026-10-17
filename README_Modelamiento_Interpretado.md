# Sistema de Detección Predictiva de Fraude en Transacciones Electrónicas

**Proyecto final | Machine Learning, reglas de riesgo y simulación operativa**

**Herramientas:** Python, Jupyter Notebook, Pandas, NumPy, Scikit-learn, Matplotlib y Seaborn.

**Antecedente:** [Proyecto Fraude Midterm](https://github.com/wilydus-png/Proyecto-Fraude-Midterm)

---

## 1. Descripción del proyecto

Este proyecto continúa el análisis de patrones transaccionales y las reglas de detección de fraude desarrolladas durante el Midterm. En esta segunda fase se entrenaron tres modelos de clasificación supervisada —Regresión Logística, Árbol de Decisión y Random Forest— y se compararon con las reglas R1, R2 y R3.

La investigación incluye preparación de datos, ingeniería de características, evaluación cronológica, simulación de umbrales, priorización de alertas mediante Top-K, una estrategia híbrida y análisis económico con supuestos hipotéticos. Su objetivo es estudiar el equilibrio entre detección de fraudes, alertas falsas y carga operativa, **no implementar un sistema de autorización en producción**.

## 2. Objetivos

**Objetivo general:** Desarrollar y evaluar modelos predictivos de detección de fraude en transacciones electrónicas, compararlos con reglas tradicionales y analizar su comportamiento operativo y económico.

**Objetivos específicos:**

1. Preparar las transacciones históricas y construir características predictivas.
2. Entrenar Regresión Logística, Árbol de Decisión y Random Forest.
3. Evaluar Precision, Recall, F1-Score, Average Precision y matrices de confusión.
4. Comparar los modelos con las reglas R1, R2 y R3 del Midterm.
5. Simular distintos umbrales, capacidades de revisión Top-K y una estrategia híbrida.
6. Estimar costos hipotéticos y analizar la sensibilidad de las decisiones.

## 3. Fuente de datos y distribución de clases

Se utilizaron los archivos `fraudTrain.csv` y `fraudTest.csv` del conjunto **Credit Card Transactions Fraud Detection**, disponible en Kaggle. La etiqueta objetivo es `is_fraud`: **0 = legítima** y **1 = fraudulenta**.

| Conjunto original | Transacciones | Fraudes | Tasa de fraude |
|---|---:|---:|---:|
| `fraudTrain.csv` | 1.296.675 | 7.506 | 0,579 % |
| `fraudTest.csv` | 555.719 | 2.145 | 0,386 % |
| **Total** | **1.852.394** | **9.651** | **0,521 %** |

### Gráfico 1. Distribución y desbalance de clases

![Desbalance de clases](graficos/01_desbalance_clases.png)

**Interpretación técnica:** La proporción de fraude es muy reducida: aproximadamente 6 transacciones fraudulentas por cada 1.000 en el archivo de entrenamiento original y 4 por cada 1.000 en el archivo de prueba. Esto significa que un clasificador que predijera siempre «legítima» conseguiría una exactitud superior al 99 %, pero no detectaría ningún fraude. Por esa razón, el proyecto prioriza Recall, Precision, F1-Score y Average Precision por encima de Accuracy. Además, la diferencia de prevalencia entre periodos exige cautela al comparar la precisión de las alertas.

**Implicación:** La evaluación de un sistema antifraude debe prestar atención a la clase minoritaria y a la carga de falsas alertas, no solo al porcentaje total de aciertos.

## 4. Metodología

### 4.1. Preparación e ingeniería de características

Se verificaron las columnas y valores nulos, se convirtió la fecha de transacción a formato temporal, se extrajeron características de horario y se calculó la distancia geográfica aproximada mediante Haversine.

| Variable | Uso |
|---|---|
| `amt` | Monto transaccional |
| `category` | Categoría del comercio |
| `merchant` | Identificador categórico del comercio |
| `hora` | Hora de la transacción |
| `dia_semana` | Día de la semana |
| `fin_semana` | Indicador de sábado o domingo |
| `distancia_km` | Distancia entre coordenadas asociadas al cliente y al comercio |
| `city_pop` | Población de la ciudad |

La distancia geográfica es una característica del conjunto de datos: **no prueba dónde estaba físicamente el titular ni su desplazamiento real**. Se utilizó `StandardScaler` para variables numéricas y `OneHotEncoder` para variables categóricas, integrados mediante `ColumnTransformer` y `Pipeline`. Los transformadores se ajustaron solo con entrenamiento.

### 4.2. División cronológica

| Conjunto | Transacciones | Fraudes | Uso |
|---|---:|---:|---|
| Entrenamiento | 1.037.340 | 5.968 | Aprender patrones |
| Validación | 259.335 | 1.538 | Analizar umbrales y escenarios |
| Prueba final | 555.719 | 2.145 | Evaluación posterior con configuraciones fijadas |

La división respeta el orden cronológico. Los resultados de validación se utilizaron para estudiar diferentes configuraciones. Los umbrales finales se fijaron antes de calcular las métricas del conjunto de prueba.

### 4.3. Modelos implementados

- **Regresión Logística:** `class_weight="balanced"`, `solver="liblinear"`, `max_iter=500`.
- **Árbol de Decisión:** `max_depth=10`, `min_samples_leaf=50`, `class_weight="balanced"`.
- **Random Forest:** `n_estimators=50`, `max_depth=12`, `min_samples_leaf=50`, `class_weight="balanced_subsample"`.

Los tres modelos utilizaron las mismas ocho variables iniciales. Se aplicaron ponderaciones para tratar el desbalance. Por ello, sus salidas de `predict_proba` se interpretan como **puntuaciones de riesgo no calibradas**, no como probabilidades exactas de fraude.

## 5. Entrenamiento y evaluación inicial de los modelos

### Gráfico 2. Matriz de confusión: Regresión Logística

![Matriz de confusión de Regresión Logística](graficos/02_matriz_confusion_logistica.png)

**Interpretación técnica:** Sobre las 259.335 transacciones de validación, la Regresión Logística con umbral 0,50 identificó **1.171 fraudes** y dejó **367 sin detectar**. Al mismo tiempo produjo **31.463 falsas alertas** y clasificó correctamente **226.334 transacciones legítimas**. Su Recall fue del **76,14 %**, pero su Precision de apenas **3,59 %**. En otras palabras, aproximadamente 96 de cada 100 alertas correspondían a operaciones legítimas.

**Implicación:** Una cobertura de fraude relativamente alta puede resultar operativamente costosa si el sistema genera demasiadas revisiones innecesarias.

### Gráfico 3. Simulación de umbrales: Regresión Logística

![Umbrales de Regresión Logística](graficos/03_umbrales_logistica.png)

**Interpretación técnica:** Elevar el umbral de 0,50 a 0,80 redujo las falsas alertas de **31.463 a 3.197** (aproximadamente **89,8 % menos**), mientras los fraudes detectados descendieron de **1.171 a 1.017**. La precisión aumentó del **3,59 % al 24,13 %** y el Recall disminuyó del **76,14 % al 66,12 %**. Se observa un compromiso entre la cantidad de operaciones investigadas y la proporción de fraudes identificados.

**Implicación:** El umbral debe elegirse según la tolerancia al fraude no detectado, la capacidad de revisión y los costos; un aumento de precisión no es necesariamente una mejora global.

### Gráfico 4. Comparación inicial: Regresión Logística vs. Árbol de Decisión

![Regresión Logística vs Árbol de Decisión](graficos/04_logistica_vs_arbol.png)

**Interpretación técnica:** En validación, con umbral 0,50 en ambos modelos, el Árbol de Decisión detectó **1.505 fraudes** frente a **1.171** de la Regresión Logística. Además, generó **9.997 falsas alertas**, mientras que la Regresión Logística produjo **31.463**. Su Recall fue del **97,85 %** y su Precision del **13,08 %**, superando a la Regresión Logística en ambas métricas bajo estas configuraciones iniciales.

**Implicación:** Un algoritmo capaz de capturar relaciones no lineales puede mejorar el desempeño frente a un modelo lineal, aunque la comparación de umbrales fijos no equivale a una comparación bajo idéntica capacidad operativa.

### Gráfico 5. Comparación integral de los tres modelos

![Comparación de modelos](graficos/05_comparacion_modelos.png)

**Interpretación técnica:** En el conjunto de validación y con umbral 0,50, el Árbol de Decisión obtuvo el mayor Recall (**97,85 %**), mientras Random Forest presentó mayor Precision (**14,56 %**) y F1-Score (**24,45 %**) que los otros modelos. Random Forest también registró el mayor Average Precision (aproximadamente **0,645**), frente a **0,496** del Árbol y **0,168** de Regresión Logística. Esto sugiere una mejor capacidad general de Random Forest para ordenar operaciones según el riesgo, aunque el Árbol fue más sensible al fraude con el umbral evaluado.

**Implicación:** El modelo más útil depende de si se busca detectar la mayor cantidad de fraudes o priorizar un grupo limitado de alertas de alta calidad.

## 6. Simulación de umbrales y capacidad operativa

### Gráfico 6. Umbrales: Árbol de Decisión vs. Random Forest

![Umbrales de Árbol y Random Forest](graficos/06_umbrales_arbol_rf.png)

**Interpretación técnica:** El Árbol de Decisión mantuvo un Recall alto incluso con umbral 0,90, con **94,93 %** y **1.460 fraudes identificados**, aunque produjo **5.271 falsas alertas**. Random Forest mostró un comportamiento más selectivo: con umbral 0,70 alcanzó **71,02 % de Precision**, detectó **946 fraudes** y produjo solo **386 falsas alertas**, a costa de disminuir su Recall hasta **61,51 %**. Los valores numéricos de umbral no son directamente equivalentes entre modelos, ya que sus distribuciones de puntuaciones difieren.

**Implicación:** Es preferible comparar también los sistemas con un número igual de alertas o bajo una restricción operativa común.

### Gráfico 7. Comparación mediante Top-K

![Comparación Top-K](graficos/07_comparacion_topk.png)

Se ordenaron las transacciones de validación por puntuación de riesgo y se seleccionaron las primeras K, utilizando la misma cantidad de alertas en los tres modelos.

| Capacidad de revisión | Modelo con más fraudes detectados | Fraudes identificados | Recall@K |
|---|---|---:|---:|
| 1.000 | Random Forest | 820 | 53,32 % |
| 3.000 | Random Forest | 1.134 | 73,73 % |
| 5.000 | Árbol de Decisión | 1.294 | 84,14 % |
| 10.000 | Árbol de Decisión | 1.494 | 97,14 % |

**Interpretación técnica:** Random Forest permitió identificar más fraudes en los escenarios de **1.000 y 3.000 alertas**, mientras que el Árbol de Decisión fue superior con **5.000 y 10.000 alertas**. En el Top-1.000, Random Forest consiguió **820 fraudes y 180 falsas alertas**, equivalente a una Precision@K del **82 %**. Con 10.000 revisiones, el Árbol alcanzó una cobertura de fraude muy alta, aunque con más falsos positivos absolutos.

**Implicación:** La asignación de recursos de revisión cambia la conveniencia relativa de los modelos. Estos valores corresponden al periodo completo de validación, no a una capacidad diaria; los empates en puntuaciones también pueden afectar los rankings.

## 7. Reglas tradicionales frente a Machine Learning

Las reglas exploratorias implementadas fueron:

| Regla | Condición |
|---|---|
| R1 | `amt >= 200` |
| R2 | R1 y horario entre las 22:00 y las 03:59 |
| R3 | R2 y categoría en `shopping_net`, `misc_net` o `grocery_pos` |

> **Nota metodológica:** Las definiciones se aplicaron en el modelamiento como reproducción operativa de las reglas del Midterm. Su correspondencia literal con el Notebook original debe confirmarse antes de considerar la comparación una réplica exacta. Además, las reglas surgieron de un análisis exploratorio sobre datos históricos integrados, por lo que la comparación con los modelos es retrospectiva.

### Gráfico 8. Reglas tradicionales vs. modelos predictivos

![Reglas vs Machine Learning](graficos/08_reglas_vs_machine_learning.png)

**Interpretación técnica:** Sobre validación, R1 detectó **1.177 fraudes** con **11.217 falsas alertas**; R2 detectó **992**, con **2.629** falsas alertas; y R3 detectó **740**, con **1.329** falsas alertas. R3 fue la regla más precisa (**35,77 %**), pero su Recall fue de **48,11 %**, el menor de las tres. El Árbol de Decisión con umbral 0,50 identificó **1.505 fraudes** y obtuvo el mayor Recall (**97,85 %**), aunque su Precision fue inferior a R3.

**Implicación:** Añadir condiciones a una regla puede reducir el volumen de alertas, pero también excluir fraudes reales. Los modelos predictivos aportan opciones adicionales de priorización; no superan automáticamente a las reglas en todos los indicadores.

### 7.1. Estrategia híbrida R3 + Random Forest

En un escenario de **5.000 alertas** se priorizaron primero las operaciones que activaban R3 y después se completó la capacidad con las puntuaciones más elevadas de Random Forest. La estrategia híbrida detectó **1.158 fraudes**, frente a **1.162** de Random Forest independiente y **1.294** del Árbol de Decisión.

**Interpretación:** Forzar la prioridad de R3 no mejoró la selección de operaciones en este escenario. La combinación de reglas y modelos solo debe adoptarse cuando se demuestre un beneficio medible.

## 8. Evaluación final sobre el conjunto de prueba

Antes de evaluar las **555.719 transacciones** de prueba se fijaron estos umbrales: Regresión Logística **0,80**, Árbol de Decisión **0,90** y Random Forest **0,70**. Las reglas mantuvieron sus condiciones.

| Estrategia | Alertas | Fraudes detectados | Falsas alertas | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 25.353 | 1.611 | 23.742 | 6,35 % | 75,10 % | 11,72 % |
| R2 | 7.035 | 1.363 | 5.672 | 19,37 % | 63,54 % | 29,69 % |
| R3 | 3.898 | 1.069 | 2.829 | 27,42 % | 49,84 % | 35,38 % |
| Regresión Logística | 8.252 | 1.395 | 6.857 | 16,90 % | 65,03 % | 26,83 % |
| Árbol de Decisión | 13.254 | 2.001 | 11.253 | 15,10 % | **93,29 %** | 25,99 % |
| Random Forest | 2.077 | 1.275 | 802 | **61,39 %** | 59,44 % | **60,40 %** |

**Lectura principal:** El Árbol de Decisión maximizó la detección en prueba, mientras Random Forest obtuvo la mayor precisión entre las estrategias evaluadas con configuraciones finales. No equivalen a dos sistemas operativos con igual volumen de trabajo: sus alertas totales son distintas.

### Gráfico 9. Validación frente a prueba

![Validación vs prueba](graficos/09_validacion_vs_prueba.png)

**Interpretación técnica:** Al comparar cada estrategia utilizando las mismas condiciones y umbrales en validación y prueba, el Recall del Árbol descendió de **94,93 % a 93,29 %** y el de Random Forest de **61,51 % a 59,44 %**. Las diferencias de Recall fueron relativamente pequeñas, pero las precisiones disminuyeron en las seis estrategias. La tasa de fraude también bajó del **0,593 % en validación** al **0,386 % en prueba**, lo que puede contribuir a una menor proporción de alertas verdaderas.

**Implicación:** Mantener un Recall relativamente similar es alentador, pero no garantiza estabilidad futura. Se requieren más periodos y seguimiento de cambios en la distribución de las transacciones.

## 9. Eficiencia operativa

### Gráfico 10. Carga de revisión de alertas

![Eficiencia operativa](graficos/10_eficiencia_operativa.png)

**Interpretación técnica:** Random Forest generó aproximadamente **1,63 alertas por fraude identificado**, incluidas **0,63 falsas alertas por cada fraude detectado**. En cambio, el Árbol de Decisión generó aproximadamente **6,62 alertas por fraude** y **5,62 falsas alertas por fraude**, pero identificó **726 fraudes adicionales** en prueba. Estos indicadores describen dos perfiles distintos: alta precisión y carga limitada en Random Forest, frente a máxima cobertura con más revisiones en el Árbol.

**Implicación:** No basta con minimizar revisiones por fraude: también hay que cuantificar las pérdidas asociadas con los fraudes que cada estrategia deja pasar.

## 10. Simulación económica

Se estimó un costo simplificado:

**Costo total = (alertas × costo por revisión) + (fraudes no detectados × pérdida por fraude)**.

Los supuestos del escenario inicial fueron **USD 2 por alerta revisada** y **USD 100 por fraude no detectado**. No son costos observados de ninguna institución.

| Estrategia | Costo total hipotético |
|---|---:|
| Árbol de Decisión | **USD 40.908** |
| Random Forest | USD 91.154 |
| Regresión Logística | USD 91.504 |
| R2 | USD 92.270 |
| R1 | USD 104.106 |
| R3 | USD 115.396 |

### Gráfico 11. Costos por estrategia

![Comparación de costos](graficos/11_costos_estrategias.png)

**Interpretación técnica:** Con una revisión de USD 2 y una pérdida de USD 100 por fraude no detectado, el Árbol de Decisión presentó el menor costo total estimado: **USD 40.908**. Aunque produjo más alertas que Random Forest, sus **144 fraudes no detectados** generaron una pérdida hipotética de **USD 14.400**, inferior a los **USD 87.000** asociados a los **870 fraudes no detectados** por Random Forest. Este resultado depende del peso económico asignado a cada error.

**Implicación:** Si la pérdida por fraude no detectado es elevada frente al costo de revisión, una estrategia de mayor cobertura puede resultar económicamente preferible, aun produciendo más alertas.

### Gráfico 12. Sensibilidad de los costos

![Análisis de sensibilidad](graficos/12_sensibilidad_economica.png)

**Interpretación técnica:** Se evaluaron **25 combinaciones** entre costos de revisión de USD **1, 2, 5, 10 y 20** y pérdidas por fraude de USD **25, 50, 100, 250 y 500**. El Árbol de Decisión resultó la estrategia de menor costo en **15 escenarios (60 %)**, mientras Random Forest lo fue en **10 (40 %)**. Los resultados ilustran un cambio de preferencia: cuando la pérdida por fraude tiene mayor peso, conviene priorizar cobertura; cuando la revisión resulta comparativamente más costosa, puede prevalecer un modelo con menos alertas.

**Implicación:** Los costos operativos no son detalles secundarios: pueden cambiar la estrategia recomendada. Los 25 escenarios son hipotéticos y no permiten afirmar una ventaja económica universal.

## 11. Conclusiones

1. Los tres modelos pudieron entrenarse y evaluarse con datos transaccionales altamente desbalanceados.
2. En el conjunto de prueba, el **Árbol de Decisión** presentó el mayor Recall (**93,29 %**), detectando **2.001** fraudes.
3. **Random Forest** obtuvo la mayor Precision (**61,39 %**) con el umbral final seleccionado y registró **802 falsas alertas**.
4. La regla R3 fue selectiva, pero su mayor precisión frente a otras reglas estuvo acompañada de menor cobertura.
5. Top-K mostró que la estrategia preferida cambia según la capacidad de revisión: Random Forest destacó con capacidades pequeñas y el Árbol con capacidades mayores.
6. El híbrido R3 + Random Forest no mejoró el resultado de las estrategias individuales en el escenario de 5.000 alertas.
7. El análisis económico favoreció al Árbol bajo un supuesto de costos, pero el análisis de sensibilidad mostró que Random Forest puede resultar preferible bajo otros supuestos.
8. Una recomendación operativa responsable debe considerar conjuntamente el riesgo de fraude, las falsas alertas, la capacidad de los analistas y los costos reales.

## 12. Recomendaciones

- Evaluar Árbol de Decisión como candidato cuando el objetivo sea **maximizar la detección**.
- Evaluar Random Forest como candidato cuando sea esencial **priorizar alertas de mayor calidad** con capacidad limitada.
- Calibrar las puntuaciones antes de denominarlas probabilidades de fraude.
- Medir costos reales de revisión, pérdidas, recuperaciones y efectos sobre clientes antes de seleccionar políticas.
- Incorporar evaluaciones temporales adicionales, supervisión de posibles cambios en los datos y mecanismos de monitoreo.
- Verificar la definición exacta de las reglas originales y documentar cualquier ajuste.

## 13. Limitaciones del estudio

- Los datos son históricos y no corresponden a operaciones reales de una institución participante.
- El diseño de las reglas del Midterm se basó en exploración de datos integrados; por tanto, su evaluación posterior no constituye una validación completamente independiente de su concepción.
- Los modelos utilizan solo ocho variables iniciales y no incorporan señales transaccionales secuenciales, autenticación, dispositivos ni historial dinámico completo.
- Las puntuaciones no se calibraron como probabilidades reales de fraude.
- La simulación Top-K corresponde a periodos completos; no modela colas de alertas por hora o día. Los empates de puntuaciones pueden afectar resultados puntuales.
- Se exploraron numerosas configuraciones sobre un mismo conjunto de validación, lo que puede favorecer decisiones adaptadas a ese periodo.
- La simulación asume que una alerta genera revisión y que un fraude detectado puede gestionarse correctamente. No representa bloqueo, prevención efectiva ni recuperaciones reales.
- Los costos son supuestos académicos; no incluyen impacto reputacional, fricción sobre clientes, restricciones regulatorias ni costos integrales de operación.
- Las comparaciones finales usan umbrales seleccionados en validación, pero las estrategias generan distintos volúmenes de alerta.

## 14. Estructura del repositorio

```text
Proyecto-Fraude-Modelamiento/
├── README.md
├── Proyecto_Fraude_Modelamiento.ipynb
└── graficos/
    ├── 01_desbalance_clases.png
    ├── 02_matriz_confusion_logistica.png
    ├── 03_umbrales_logistica.png
    ├── 04_logistica_vs_arbol.png
    ├── 05_comparacion_modelos.png
    ├── 06_umbrales_arbol_rf.png
    ├── 07_comparacion_topk.png
    ├── 08_reglas_vs_machine_learning.png
    ├── 09_validacion_vs_prueba.png
    ├── 10_eficiencia_operativa.png
    ├── 11_costos_estrategias.png
    └── 12_sensibilidad_economica.png
```

Los archivos `fraudTrain.csv` y `fraudTest.csv` deben obtenerse del conjunto de datos de origen y ubicarse en la misma carpeta que el Notebook, según las rutas configuradas. Pueden no incluirse en GitHub por su tamaño.

## 15. Cómo reproducir el proyecto

1. Descargar o clonar el repositorio.
2. Instalar Python y Jupyter Notebook.
3. Instalar dependencias:

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

4. Obtener `fraudTrain.csv` y `fraudTest.csv` del dataset original.
5. Colocar ambos archivos junto a `Proyecto_Fraude_Modelamiento.ipynb`.
6. Abrir el Notebook en Jupyter y ejecutar sus celdas en orden.
7. Consultar las salidas y los PNG guardados en `graficos/`.

## 16. Referencia y finalidad

**Proyecto antecedente:** [Proyecto Fraude Midterm](https://github.com/wilydus-png/Proyecto-Fraude-Midterm).

**Finalidad:** Proyecto académico para estudiar técnicas de clasificación, evaluación de reglas de riesgo y simulación operativa. **No es un sistema de detección de fraude listo para producción.**
