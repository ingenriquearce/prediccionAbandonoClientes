from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

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
            ('cat', Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))]), COLUMNAS_CATEGORICAS),
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


def preprocesar_datos(X_train, X_test):
    preprocesador = crear_preprocesador()
    X_train_pre = preprocesador.fit_transform(X_train)
    X_test_pre = preprocesador.transform(X_test)
    return X_train_pre, X_test_pre, preprocesador


def obtener_modelos():
    return {
        'Regresión Logística': LogisticRegression(max_iter=1000),
        'Bosques Aleatorios': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    }


def comparar_modelos(df):
    X_train, X_test, y_train, y_test, _, _ = preparar_datos(df)
    X_train_pre, X_test_pre, _ = preprocesar_datos(X_train, X_test)

    resultados = []
    for nombre, modelo in obtener_modelos().items():
        modelo.fit(X_train_pre, y_train)
        predicciones = modelo.predict(X_test_pre)
        precision = accuracy_score(y_test, predicciones) * 100
        resultados.append({'Modelo': nombre, 'Precision': round(precision, 2)})

    return pd.DataFrame(resultados).sort_values(by='Precision', ascending=False)


def entrenar_modelo(df, nombre_modelo):
    modelos = obtener_modelos()
    modelo = modelos[nombre_modelo]

    X_train, X_test, y_train, y_test, info_train, info_test = preparar_datos(df)
    X_train_pre, X_test_pre, preprocesador = preprocesar_datos(X_train, X_test)

    modelo.fit(X_train_pre, y_train)
    predicciones = modelo.predict(X_test_pre)
    probabilidades = modelo.predict_proba(X_test_pre)[:, 1]

    resultados_clientes = info_test.copy()
    resultados_clientes['ProbabilidadAbandono'] = probabilidades
    resultados_clientes = resultados_clientes.sort_values(by='ProbabilidadAbandono', ascending=False)

    matriz = confusion_matrix(y_test, predicciones)

    return {
        'modelo': modelo,
        'preprocesador': preprocesador,
        'accuracy': accuracy_score(y_test, predicciones) * 100,
        'classification_report': classification_report(y_test, predicciones),
        'confusion_matrix': matriz,
        'cross_val_mean': cross_val_score(modelo, X_train_pre, y_train, cv=5).mean(),
        'top_clientes': resultados_clientes,
    }


def guardar_modelo(modelo, preprocesador, nombre_archivo='modelo_churn'):
    CARPETA_MODELOS.mkdir(exist_ok=True)
    joblib.dump(modelo, CARPETA_MODELOS / f'{nombre_archivo}.joblib')
    joblib.dump(preprocesador, CARPETA_MODELOS / f'{nombre_archivo}_preprocesador.joblib')
