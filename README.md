# Predicción de Abandono de Clientes

Proyecto simple y legible de machine learning con Streamlit para predecir clientes con mayor probabilidad de abandono.

## Qué hace ahora
Además del análisis y entrenamiento, esta versión ya incluye una capa básica de MLOps local:

- comparación rápida de modelos,
- ejecución de experimentos,
- guardado de artefactos por corrida,
- registro de métricas,
- historial de experimentos,
- promoción automática de un **modelo champion** si supera al anterior.

## Estructura del proyecto

```text
PrediccionAbandonoClientes/
├── app.py
├── src/
│   ├── datos.py
│   ├── analisis.py
│   ├── modelos.py
│   ├── experimentos.py
│   └── registro_modelos.py
├── data/
│   └── Churn_Data.csv
├── assets/
│   └── dataframeExample.png
├── notebooks/
│   └── PrediccionAbandonoDeClientes.ipynb
├── mlops/
│   ├── experimentos/
│   └── registry/
└── requirements.txt
```

## Flujo de trabajo

1. Cargas el dataset base o subes tu CSV.
2. La app limpia duplicados y muestra análisis exploratorio.
3. Puedes comparar varios modelos rápidamente.
4. Ejecutas un experimento con el modelo que elijas.
5. La app guarda:
   - `pipeline.joblib`
   - `metricas.json`
   - `parametros.json`
   - `metadata.json`
6. El nuevo experimento se compara contra el **champion** actual usando `f1_churn`.
7. Si gana, se promueve automáticamente como nuevo modelo productivo.

## Criterio de promoción
La métrica principal del registro es:

- `f1_churn`

Eso evita depender solo de accuracy, que puede engañar en problemas de churn.

## Qué se guarda en `mlops/`

### `mlops/experimentos/`
Cada ejecución crea una carpeta propia con sus artefactos.

### `mlops/registry/champion.json`
Guarda el modelo vigente que ganó la comparación.

### `mlops/registry/historial_modelos.csv`
Guarda el historial de experimentos registrados.

## Modelos incluidos
- Regresión Logística
- Bosques Aleatorios
- SVM
- Gradient Boosting

## Detalle técnico importante
El proyecto no usa `ClienteId`, `Apellido` ni `NroFila` para entrenar.

Eso es intencional: son identificadores, no variables de negocio útiles para aprender patrones reales de abandono.

## Cómo ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Nota sobre el guardado del modelo
Ahora se guarda un único archivo `pipeline.joblib` por experimento.

Ese pipeline incluye:
- el preprocesamiento,
- y el modelo entrenado.

Eso simplifica mucho la reutilización y reduce errores al predecir.
