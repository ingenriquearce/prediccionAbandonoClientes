<<<<<<< HEAD
# Predicción de Abandono de Clientes - Versión Refactorizada Simple

Esta versión mantiene la idea original del proyecto, pero con una arquitectura mucho más simple y legible.

## Objetivo
Predecir qué clientes tienen mayor probabilidad de abandonar el banco y mostrar los casos de mayor riesgo en una app hecha con Streamlit.

## Estructura del proyecto

```text
PrediccionAbandonoClientes_refactor_simple/
├── app.py                # Interfaz Streamlit
├── src/
│   ├── datos.py          # Carga, renombrado y limpieza
│   ├── analisis.py       # Gráficos del análisis exploratorio
│   └── modelos.py        # Preprocesamiento, entrenamiento y evaluación
├── data/
│   └── Churn_Data.csv    # Dataset base
├── assets/
│   └── dataframeExample.png
├── notebooks/
│   └── PrediccionAbandonoDeClientes.ipynb
└── modelos_guardados/    # Aquí se guardan modelo y preprocesador
```

## Qué hace cada archivo

### `app.py`
Solo controla la app:
- carga datos,
- muestra análisis exploratorio,
- compara modelos,
- entrena un modelo,
- muestra clientes con mayor riesgo,
- permite buscar un cliente.

### `src/datos.py`
Tiene lo básico para trabajar con el dataset:
- cargar el CSV base,
- cargar un CSV subido,
- renombrar columnas a español,
- eliminar duplicados.

### `src/analisis.py`
Contiene los gráficos del análisis exploratorio:
- edad,
- saldo,
- geografía,
- género,
- matriz de correlación.

### `src/modelos.py`
Aquí está la parte importante de machine learning:
- separar X e y,
- train/test split,
- escalado y one-hot encoding,
- entrenamiento de modelos,
- evaluación,
- ranking de clientes por probabilidad de abandono,
- guardado del modelo entrenado.

## Mejora importante frente al proyecto original
En esta versión **no se usan `ClienteId`, `Apellido` ni `NroFila` para entrenar**.

Eso es importante porque esas columnas son identificadores, no variables de negocio útiles para que el modelo aprenda patrones reales.

## Cómo ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Modelos incluidos
- Regresión Logística
- Bosques Aleatorios
- SVM
- Gradient Boosting

## Dónde se guarda el entrenamiento
Si presionas **Guardar modelo entrenado**, se crean archivos `.joblib` dentro de `modelos_guardados/`.

- `modelo_churn.joblib` → modelo entrenado
- `modelo_churn_preprocesador.joblib` → preprocesador ajustado

Eso permite reutilizar el entrenamiento después.
=======
# Prediccion del Abandono de Clientes
Predice el abandono de clientes en una entidad bancaria
>>>>>>> 244894ecaf43f72db7e392e7a115811848a3edc6
