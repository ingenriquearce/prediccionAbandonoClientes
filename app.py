import pandas as pd
import streamlit as st
from src.analisis import (
    grafico_edad,
    grafico_genero,
    grafico_geografia,
    grafico_saldo,
    matriz_correlacion,
    obtener_columnas_categoricas,
)
from src.datos import cargar_csv_subido, cargar_dataset_base, limpiar_datos
from src.modelos import comparar_modelos, ejecutar_experimento, entrenar_modelo, guardar_modelo
from src.registro_modelos import cargar_champion_actual, obtener_historial_experimentos

st.set_page_config(page_title='Predicción de Abandono', layout='wide')
st.title('Software de Predicción del Abandono de Clientes')


def cargar_datos_desde_interfaz():
    st.header('1. Cargar datos')
    opcion = st.radio(
        'Selecciona el origen de los datos:',
        ['Usar dataset de ejemplo', 'Subir mi archivo CSV'],
        horizontal=True,
    )

    st.write('El archivo debe tener la misma estructura del dataset original.')
    st.image('assets/dataframeExample.png', use_column_width=True)

    if opcion == 'Usar dataset de ejemplo':
        df = cargar_dataset_base()
        st.success('Se cargó el dataset de ejemplo.')
        return df

    uploaded_file = st.file_uploader('Selecciona un archivo CSV', type=['csv'])
    if uploaded_file is not None:
        try:
            df = cargar_csv_subido(uploaded_file)
            st.success('¡Carga de datos exitosa!')
            return df
        except Exception as e:
            st.error(f'Error al cargar el archivo: {e}')

    return None


def mostrar_analisis_exploratorio(df):
    st.header('2. Análisis exploratorio de datos')

    df_limpio, filas_eliminadas = limpiar_datos(df)
    st.write(f'Filas originales: {len(df)}')
    st.write(f'Filas después de eliminar duplicados: {len(df_limpio)}')
    st.write(f'Duplicados eliminados: {filas_eliminadas}')

    st.subheader('Tipos de datos')
    st.write(df_limpio.dtypes)

    st.subheader('Variables categóricas')
    st.write(obtener_columnas_categoricas(df_limpio))

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(grafico_edad(df_limpio))
        st.pyplot(grafico_geografia(df_limpio))
    with col2:
        st.pyplot(grafico_saldo(df_limpio))
        st.pyplot(grafico_genero(df_limpio))

    st.subheader('Matriz de correlación')
    st.pyplot(matriz_correlacion(df_limpio))

    return df_limpio


def mostrar_champion_actual():
    st.header('3. Modelo champion actual')
    champion = cargar_champion_actual()

    if champion is None:
        st.info('Aún no existe un modelo champion. Ejecuta un experimento para crear el primero.')
        return

    col1, col2, col3 = st.columns(3)
    col1.metric('Modelo', champion['modelo'])
    col2.metric('F1 churn', f"{champion['f1_churn']:.4f}")
    col3.metric('Recall churn', f"{champion['recall_churn']:.4f}")

    st.write(f"Run ID: `{champion['run_id']}`")
    st.write(f"Ruta del pipeline productivo: `{champion['ruta_pipeline']}`")
    st.write(f"Entrenado el: {champion['fecha_entrenamiento']}")


def mostrar_comparacion_modelos(df):
    st.header('4. Comparación rápida de modelos')
    if st.button('Comparar modelos'):
        resultados = comparar_modelos(df)
        st.dataframe(resultados, use_container_width=True)
        st.bar_chart(resultados.set_index('Modelo')['F1 churn'])


def mostrar_entrenamiento(df):
    st.header('5. Ejecutar experimento y evaluar promoción')
    nombre_modelo = st.selectbox(
        'Elige un modelo',
        ['Regresión Logística', 'Bosques Aleatorios', 'SVM', 'Gradient Boosting']
    )
    top_n = st.slider('Cantidad de clientes a mostrar', min_value=1, max_value=30, value=10)

    if st.button('Ejecutar experimento'):
        resultados = ejecutar_experimento(df, nombre_modelo)
        metricas = resultados['metricas']
        decision = resultados['decision_promocion']

        st.write(f"Run ID generado: `{resultados['run_id']}`")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Accuracy', f"{metricas['accuracy']:.4f}")
        col2.metric('F1 churn', f"{metricas['f1_churn']:.4f}")
        col3.metric('Recall churn', f"{metricas['recall_churn']:.4f}")
        col4.metric('ROC AUC', f"{metricas['roc_auc']:.4f}")

        st.write(f"Validación cruzada promedio (F1): {metricas['cross_val_mean']:.4f}")

        if decision['fue_promovido']:
            st.success(f"Modelo promovido a champion. {decision['motivo']}")
        else:
            st.warning(f"Modelo registrado, pero no promovido. {decision['motivo']}")

        st.subheader('Reporte de clasificación')
        st.text(resultados['classification_report'])

        st.subheader('Matriz de confusión')
        matriz = resultados['confusion_matrix']
        st.write(pd.DataFrame(matriz, index=['Real 0', 'Real 1'], columns=['Pred 0', 'Pred 1']))

        st.subheader(f'Top {top_n} clientes con mayor probabilidad de abandono')
        top_clientes = resultados['top_clientes'].head(top_n)
        st.dataframe(top_clientes, use_container_width=True)
        st.bar_chart(top_clientes.set_index('ClienteId')['ProbabilidadAbandono'])

        st.subheader('Artefactos generados')
        st.json(resultados['artefactos'])


def mostrar_historial_experimentos():
    st.header('6. Historial reciente de experimentos')
    historial = obtener_historial_experimentos(limit=10)

    if historial.empty:
        st.info('Todavía no hay experimentos registrados.')
        return

    st.dataframe(historial, use_container_width=True)


def mostrar_busqueda_cliente(df):
    st.header('7. Búsqueda de cliente por ID')
    cliente_id = st.text_input('Introduce el ID del cliente')

    if st.button('Buscar cliente'):
        if not cliente_id.strip():
            st.warning('Escribe un ClienteId.')
            return

        try:
            cliente_id = int(cliente_id)
        except ValueError:
            st.warning('El ClienteId debe ser numérico.')
            return

        cliente = df[df['ClienteId'] == cliente_id]
        if cliente.empty:
            st.warning(f'Cliente con ClienteId {cliente_id} no encontrado.')
        else:
            st.dataframe(cliente, use_container_width=True)


def main():
    df = cargar_datos_desde_interfaz()
    if df is None:
        st.info('Carga un dataset para continuar.')
        return

    st.subheader('Vista previa de los datos')
    st.dataframe(df.head(), use_container_width=True)

    df_limpio = mostrar_analisis_exploratorio(df)
    mostrar_champion_actual()
    mostrar_comparacion_modelos(df_limpio)
    mostrar_entrenamiento(df_limpio)
    mostrar_historial_experimentos()
    mostrar_busqueda_cliente(df_limpio)


if __name__ == '__main__':
    main()
