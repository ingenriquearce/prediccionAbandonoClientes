import streamlit as st
import pandas as pd
from src.datos import cargar_csv_subido, cargar_dataset_base, limpiar_datos
from src.analisis import (
    obtener_columnas_categoricas,
    grafico_edad,
    grafico_saldo,
    grafico_geografia,
    grafico_genero,
    matriz_correlacion,
)
from src.modelos import comparar_modelos, entrenar_modelo, guardar_modelo

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


def mostrar_comparacion_modelos(df):
    st.header('3. Comparación de modelos')
    if st.button('Comparar modelos'):
        resultados = comparar_modelos(df)
        st.dataframe(resultados, use_container_width=True)
        st.bar_chart(resultados.set_index('Modelo'))


def mostrar_entrenamiento(df):
    st.header('4. Entrenar un modelo y ver clientes con mayor riesgo')
    nombre_modelo = st.selectbox(
        'Elige un modelo',
        ['Regresión Logística', 'Bosques Aleatorios', 'SVM', 'Gradient Boosting']
    )
    top_n = st.slider('Cantidad de clientes a mostrar', min_value=1, max_value=30, value=10)

    if st.button('Entrenar modelo seleccionado'):
        resultados = entrenar_modelo(df, nombre_modelo)

        st.success(f"Precisión del modelo: {resultados['accuracy']:.2f}%")
        st.write(f"Precisión promedio en validación cruzada: {resultados['cross_val_mean']:.2f}")

        st.subheader('Reporte de clasificación')
        st.text(resultados['classification_report'])

        st.subheader('Matriz de confusión')
        matriz = resultados['confusion_matrix']
        st.write(pd.DataFrame(matriz, index=['Real 0', 'Real 1'], columns=['Pred 0', 'Pred 1']))

        st.subheader(f'Top {top_n} clientes con mayor probabilidad de abandono')
        top_clientes = resultados['top_clientes'].head(top_n)
        st.dataframe(top_clientes, use_container_width=True)
        st.bar_chart(top_clientes.set_index('ClienteId')['ProbabilidadAbandono'])

        if st.button('Guardar modelo entrenado'):
            guardar_modelo(resultados['modelo'], resultados['preprocesador'])
            st.success('Modelo y preprocesador guardados en la carpeta modelos_guardados.')


def mostrar_busqueda_cliente(df):
    st.header('5. Búsqueda de cliente por ID')
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
    mostrar_comparacion_modelos(df_limpio)
    mostrar_entrenamiento(df_limpio)
    mostrar_busqueda_cliente(df_limpio)


if __name__ == '__main__':
    main()
