from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from src.experimentos import guardar_experimento, generar_run_id, registrar_en_historial
from src.registro_modelos import evaluar_promocion

BASE_DIR = Path(__file__).resolve().parent.parent
CARPETA_MODELOS = BASE_DIR / 'modelos_guardados'

COLUMNAS_NUMERICAS = [
    'PuntuacionCredito', 'Edad', 'Tenencia', 'Saldo',
    'NroProductos', 'TieneTarjetaCredito', 'EsMiembroActivo', 'SalarioEstimado'
]
COLUMNAS_CATEGORICAS = ['Geografia', 'Genero']
COLUMNAS_INFO = ['ClienteId', 'Apellido']
COLUMNA_OBJETIVO = 'Churn'


def crear_preprocesador():
    return ColumnTransformer(
        transformers=[
            ('num', Pipeline([('scaler', StandardScaler())]), COLUMNAS_NUMERICAS),
            (
                'cat',
                Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]),
                COLUMNAS_CATEGORICAS,
            ),
        ]
    )


def preparar_datos(df):
    """
    Separa el dataset en entrenamiento y prueba.
    No usamos ClienteId, NroFila ni Apellido como variables del modelo,
    porque son identificadores y no aportan al aprendizaje real.
    """
    columnas_modelo = COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS
    X = df[columnas_modelo]
    y = df[COLUMNA_OBJETIVO]
    info_clientes = df[COLUMNAS_INFO]

    return train_test_split(
        X,
        y,
        info_clientes,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )


def obtener_modelos():
    return {
        'Regresión Logística': LogisticRegression(max_iter=1000, random_state=42),
        'Bosques Aleatorios': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    }


def crear_pipeline(nombre_modelo: str) -> Pipeline:
    modelos = obtener_modelos()
    modelo = modelos[nombre_modelo]
    return Pipeline(
        [
            ('preprocesador', crear_preprocesador()),
            ('modelo', modelo),
        ]
    )


def calcular_metricas(y_test, predicciones, probabilidades, pipeline, X_train, y_train):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predicciones,
        labels=[1],
        average=None,
        zero_division=0,
    )

    return {
        'accuracy': float(accuracy_score(y_test, predicciones)),
        'precision_churn': float(precision[0]),
        'recall_churn': float(recall[0]),
        'f1_churn': float(f1[0]),
        'roc_auc': float(roc_auc_score(y_test, probabilidades)),
        'cross_val_mean': float(cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1').mean()),
    }


def comparar_modelos(df):
    X_train, X_test, y_train, y_test, _, _ = preparar_datos(df)

    resultados = []
    for nombre in obtener_modelos().keys():
        pipeline = crear_pipeline(nombre)
        pipeline.fit(X_train, y_train)
        predicciones = pipeline.predict(X_test)
        probabilidades = pipeline.predict_proba(X_test)[:, 1]

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            predicciones,
            labels=[1],
            average=None,
            zero_division=0,
        )

        resultados.append(
            {
                'Modelo': nombre,
                'Accuracy': round(float(accuracy_score(y_test, predicciones)), 4),
                'F1 churn': round(float(f1[0]), 4),
                'Recall churn': round(float(recall[0]), 4),
                'ROC AUC': round(float(roc_auc_score(y_test, probabilidades)), 4),
            }
        )

    return pd.DataFrame(resultados).sort_values(by='F1 churn', ascending=False)


def ejecutar_experimento(df, nombre_modelo):
    X_train, X_test, y_train, y_test, _, info_test = preparar_datos(df)
    pipeline = crear_pipeline(nombre_modelo)

    pipeline.fit(X_train, y_train)
    predicciones = pipeline.predict(X_test)
    probabilidades = pipeline.predict_proba(X_test)[:, 1]

    metricas = calcular_metricas(y_test, predicciones, probabilidades, pipeline, X_train, y_train)
    matriz = confusion_matrix(y_test, predicciones)
    reporte_texto = classification_report(y_test, predicciones, zero_division=0)

    resultados_clientes = info_test.copy()
    resultados_clientes['ProbabilidadAbandono'] = probabilidades
    resultados_clientes = resultados_clientes.sort_values(by='ProbabilidadAbandono', ascending=False)

    run_id = generar_run_id(nombre_modelo)
    parametros = {
        'modelo': pipeline.named_steps['modelo'].__class__.__name__,
        'nombre_modelo_interfaz': nombre_modelo,
        'columnas_numericas': COLUMNAS_NUMERICAS,
        'columnas_categoricas': COLUMNAS_CATEGORICAS,
    }
    metadata = {
        'dataset': 'dataset_cargado_en_streamlit',
        'n_filas': int(len(df)),
        'n_columnas': int(df.shape[1]),
        'fecha_entrenamiento': datetime.now().isoformat(timespec='seconds'),
        'columna_objetivo': COLUMNA_OBJETIVO,
        'columnas_modelo': COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS,
    }

    artefactos = guardar_experimento(run_id, pipeline, metricas, parametros, metadata)
    decision = evaluar_promocion(run_id, nombre_modelo, metricas, artefactos['pipeline_path'], metadata)
    registrar_en_historial(run_id, nombre_modelo, metricas, decision['fue_promovido'])

    return {
        'run_id': run_id,
        'pipeline': pipeline,
        'metricas': metricas,
        'classification_report': reporte_texto,
        'confusion_matrix': matriz,
        'top_clientes': resultados_clientes,
        'artefactos': artefactos,
        'decision_promocion': decision,
    }


def guardar_modelo(pipeline, nombre_archivo='modelo_churn'):
    CARPETA_MODELOS.mkdir(exist_ok=True)
    joblib.dump(pipeline, CARPETA_MODELOS / f'{nombre_archivo}.joblib')
