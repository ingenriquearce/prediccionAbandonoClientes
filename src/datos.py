from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Churn_Data.csv"

COLUMNAS_ES = [
    'NroFila', 'ClienteId', 'Apellido', 'PuntuacionCredito', 'Geografia',
    'Genero', 'Edad', 'Tenencia', 'Saldo', 'NroProductos',
    'TieneTarjetaCredito', 'EsMiembroActivo', 'SalarioEstimado', 'Churn'
]

MAPEO_INGLES_A_ES = {
    'RowNumber': 'NroFila',
    'CustomerId': 'ClienteId',
    'Surname': 'Apellido',
    'CreditScore': 'PuntuacionCredito',
    'Geography': 'Geografia',
    'Gender': 'Genero',
    'Age': 'Edad',
    'Tenure': 'Tenencia',
    'Balance': 'Saldo',
    'NumOfProducts': 'NroProductos',
    'HasCrCard': 'TieneTarjetaCredito',
    'IsActiveMember': 'EsMiembroActivo',
    'EstimatedSalary': 'SalarioEstimado',
    'Exited': 'Churn',
}


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Deja las columnas en español para usar el resto del proyecto."""
    df = df.copy()

    # Caso 1: viene con columnas en inglés como el dataset original.
    if set(MAPEO_INGLES_A_ES.keys()).issubset(df.columns):
        return df.rename(columns=MAPEO_INGLES_A_ES)

    # Caso 2: viene con 14 columnas y se las renombramos en el orden esperado.
    if len(df.columns) == len(COLUMNAS_ES):
        df.columns = COLUMNAS_ES
        return df

    raise ValueError(
        "El archivo debe tener 14 columnas y la misma estructura del dataset de churn."
    )


def cargar_dataset_base() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return normalizar_columnas(df)


def cargar_csv_subido(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    return normalizar_columnas(df)


def limpiar_datos(df: pd.DataFrame):
    """Elimina duplicados y devuelve el dataframe limpio + cuántas filas se quitaron."""
    df_limpio = df.drop_duplicates().copy()
    filas_eliminadas = len(df) - len(df_limpio)
    return df_limpio, filas_eliminadas
