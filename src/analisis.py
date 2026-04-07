import matplotlib.pyplot as plt

COLUMNAS_NUMERICAS = [
    'PuntuacionCredito', 'Edad', 'Tenencia', 'Saldo',
    'NroProductos', 'TieneTarjetaCredito', 'EsMiembroActivo', 'SalarioEstimado'
]


def obtener_columnas_categoricas(df):
    return df.select_dtypes(include=['object']).columns.tolist()


def grafico_edad(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df['Edad'].astype(float), bins=20, edgecolor='black')
    ax.set_title('Distribución de Edad')
    ax.set_xlabel('Edad')
    ax.set_ylabel('Frecuencia')
    return fig


def grafico_saldo(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df['Saldo'], bins=20, edgecolor='black')
    ax.set_title('Distribución de Saldo')
    ax.set_xlabel('Saldo')
    ax.set_ylabel('Frecuencia')
    return fig


def grafico_geografia(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    conteo = df['Geografia'].value_counts()
    ax.bar(conteo.index, conteo.values)
    ax.set_title('Distribución de Geografía')
    ax.set_ylabel('Cantidad')
    return fig


def grafico_genero(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    conteo = df['Genero'].value_counts()
    ax.bar(conteo.index, conteo.values)
    ax.set_title('Distribución de Género')
    ax.set_ylabel('Cantidad')
    return fig


def matriz_correlacion(df):
    corr = df[COLUMNAS_NUMERICAS].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    imagen = ax.imshow(corr, cmap='coolwarm', interpolation='nearest')
    fig.colorbar(imagen)
    ax.set_title('Matriz de Correlación')
    ax.set_xticks(range(len(COLUMNAS_NUMERICAS)))
    ax.set_xticklabels(COLUMNAS_NUMERICAS, rotation=45, ha='right')
    ax.set_yticks(range(len(COLUMNAS_NUMERICAS)))
    ax.set_yticklabels(COLUMNAS_NUMERICAS)
    return fig
