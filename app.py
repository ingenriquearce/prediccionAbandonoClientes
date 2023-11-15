import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

data = None

# Nombre de la aplicación
st.title('Software de Predicción del Abandono de Clientes')

def cargar_datos():
    st.header("Cargar los Datos")   
    st.write('En este orden, con esos nombres y con esos tipos de datos, deben estar las columnas')
    st.image('dataframeExample.png', use_column_width=True)
    uploaded_file = st.file_uploader("Selecciona un archivo CSV con el conjunto de Datos", type=["csv"])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            # Renombrar las columnas
            nuevos_nombres = {
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
                'Exited': 'Churn'
            }
            data = data.rename(columns=nuevos_nombres)
            return data # Retorna el dataframe
        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")
    return None

data = cargar_datos()

def limpieza():
    # Eliminar duplicados en todo el DataFrame
    data_sin_duplicados = data.drop_duplicates()

    # Calcular el número de filas eliminadas (datos duplicados)
    filas_eliminadas = len(data) - len(data_sin_duplicados)

    # Crear una figura con dos subtramas
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)

    # Gráfico antes de eliminar duplicados
    plt.bar(['Antes'], [len(data)], color='lightblue', label='Antes de Eliminar Duplicados')
    plt.ylabel('Cantidad de Filas')
    plt.title('Cantidad de Filas Antes de Eliminar Duplicados')

    plt.subplot(1, 2, 2)

    # Gráfico después de eliminar duplicados
    plt.bar(['Después'], [len(data_sin_duplicados)], color='salmon', label='Después de Eliminar Duplicados')
    plt.ylabel('Cantidad de Filas')
    plt.title('Cantidad de Filas Después de Eliminar Duplicados')

    plt.tight_layout()
    st.write("¡Datos Limpios!")

    if filas_eliminadas > 0:
        # Mostrar los gráficos
        st.pyplot(plt)
        st.write(f'Filas eliminadas: {filas_eliminadas}')

def analisisExploratorio():
    st.header("Análisis Exploratorio de Datos")
    
    # Definir columnas numéricas
    numeric_features = ['PuntuacionCredito', 'Edad', 'Tenencia', 'Saldo', 'NroProductos', 'TieneTarjetaCredito', 'EsMiembroActivo', 'SalarioEstimado']

    # Mostrar tipos de datos y variables categóricas
    st.subheader("Tipos de Datos:")
    st.write(data.dtypes)

    st.subheader("Variables Categóricas:")
    st.write(data.select_dtypes(include=['object']).columns.tolist())

    # Distribución de Edad
    st.subheader("Distribución de Edad:")
    
    # Convertir la columna 'Edad' a tipo float
    data['Edad'] = data['Edad'].astype(float)

    # Crear un histograma de la variable 'Edad'
    plt.hist(data['Edad'], bins=20, color='lightblue', edgecolor='black')
    plt.xlabel('Edad')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Edad')
    
    # Mostrar la figura en Streamlit
    st.pyplot(plt)
    
    # Distribución de Balance
    st.subheader("Distribución de Balance:")
    
    # Crear un histograma de la variable 'Saldo'
    plt.hist(data['Saldo'], bins=20, color='lightblue', edgecolor='black')
    plt.xlabel('Saldo')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Saldo')
    
    # Mostrar la figura en Streamlit
    st.pyplot(plt)

    # Visualización de Geografía
    st.subheader("Distribución de Geografía:")
    st.bar_chart(data['Geografia'].value_counts())

    # Visualización de Género
    st.subheader("Distribución de Género:")
    st.bar_chart(data['Genero'].value_counts())

    st.subheader("Distribución de NroProductos:")
    st.write(data['NroProductos'].value_counts())

    st.subheader("Distribución de TieneTarjetaCredito:")
    st.write(data['TieneTarjetaCredito'].value_counts())

    # Matriz de correlación de características numéricas
    st.subheader("Matriz de Correlación de Características Numéricas:")
    correlation_matrix = data[numeric_features].corr()
    
    # Crear un mapa de calor
    plt.figure(figsize=(8, 6))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title("Matriz de Correlación")
    plt.xticks(range(len(numeric_features)), numeric_features, rotation=45)
    plt.yticks(range(len(numeric_features)), numeric_features)
    st.pyplot(plt)

    # Crear un gráfico de barras para mostrar valores atípicos identificados
    initial_outliers = (data[numeric_features] > 3).sum()
    st.subheader("Valores Atípicos Identificados por Columna:")
    st.bar_chart(initial_outliers)

def preprocesamiento():
    # Definir columnas numéricas y categóricas
    numeric_features = ['PuntuacionCredito', 'Edad', 'Tenencia', 'Saldo', 'NroProductos', 'TieneTarjetaCredito', 'EsMiembroActivo', 'SalarioEstimado']
    categorical_features = ['Geografia', 'Genero']

    # Separar características (X) y variable objetivo (y)
    X = data.drop(columns=['Churn'])
    y = data['Churn']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear un transformador para aplicar escalado a las variables numéricas
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Crear un transformador para aplicar codificación one-hot a las variables categóricas
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Crear una transformación de columnas combinando los transformadores numéricos y categóricos
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Aplicar la transformación a los datos de entrenamiento y prueba
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Obtener las categorías únicas de las columnas categóricas después de la codificación one-hot
    categorical_encoder = preprocessor.named_transformers_['cat']['onehot']
    categorical_column_names = categorical_encoder.get_feature_names_out(input_features=categorical_features)

    # Unir los nombres de columnas numéricas y categóricas después de la transformación
    feature_names = numeric_features + list(categorical_column_names)

    # Crear un DataFrame con las características preprocesadas y los nombres de las columnas
    X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed, columns=feature_names)

    st.write("¡Preprocesamiento de Datos Exitoso!")

    return X_train_preprocessed, X_test_preprocessed, X_train, X_test, y_train, y_test

def seleccionModelo():
    X_train_preprocessed, X_test_preprocessed, X_train, X_test, y_train, y_test = preprocesamiento()

    # Entrenar los dos modelos
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train_preprocessed, y_train)
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train_preprocessed, y_train)
    svm_model = SVC(probability=True)
    svm_model.fit(X_train_preprocessed, y_train)
    gb_model = GradientBoostingClassifier()
    gb_model.fit(X_train_preprocessed, y_train)
    # Predecir en el conjunto de prueba
    y_pred_logistic = logistic_model.predict(X_test_preprocessed)
    y_pred_rf = rf_model.predict(X_test_preprocessed)
    y_pred_svm = svm_model.predict(X_test_preprocessed)
    y_pred_gb = gb_model.predict(X_test_preprocessed)
    # Resultados de precisión de los modelos
    accuracy_logistic = accuracy_score(y_test, y_pred_logistic) * 100
    accuracy_rf = accuracy_score(y_test, y_pred_rf) * 100
    accuracy_svm = accuracy_score(y_test, y_pred_svm) * 100
    accuracy_gb = accuracy_score(y_test, y_pred_gb) * 100

    accuracy_scores = [accuracy_logistic, accuracy_rf, accuracy_svm, accuracy_gb]
    models = ['Regresión Logística', 'Bosques Aleatorios', 'SVM', 'GradientBoost']

    # Crear una gráfica de barras en Streamlit
    st.title('Precisión de Modelos')
    st.bar_chart(dict(zip(models, accuracy_scores)))

st.set_option('deprecation.showPyplotGlobalUse', False)
def entrenamientoRegresion(cant_top_clientes):
    preprocesamiento()
    X_train_preprocessed, X_test_preprocessed, X_train, X_test, y_train, y_test = preprocesamiento()
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Crear y entrenar el modelo seleccionado (Regresión Logística)
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train_preprocessed, y_train)

    # Predecir en el conjunto de prueba
    y_pred = logistic_model.predict(X_test_preprocessed)

    # Calcular la precisión en el conjunto de prueba
    accuracy_logistic = accuracy_score(y_test, y_pred) * 100
    
    # Mostrar la precisión en la interfaz
    st.write(f'Precisión del modelo en el conjunto de prueba: {accuracy_logistic:.2f}%')

    # Obtener las probabilidades de abandono para cada cliente
    y_probs = logistic_model.predict_proba(X_test_preprocessed)[:, 1]

    # Crear un DataFrame con los resultados
    clientes_probables_abandono = pd.DataFrame({'Apellido': X_test['Apellido'], 'ClienteId': X_test['ClienteId'],'Probabilidad_Abandono_LR': y_probs})

    # Ordenar el DataFrame por las probabilidades en orden descendente
    clientes_probables_abandono = clientes_probables_abandono.sort_values(by='Probabilidad_Abandono_LR', ascending=False)

    # Validar que la cantidad ingresada sea válida
    if cant_top_clientes <= 0:
        st.warning("La cantidad ingresada debe ser un número positivo.")
    else:
        # Tomar solo la cantidad especificada de clientes
        top_clientes_lr = clientes_probables_abandono.head(cant_top_clientes)

        # Crear una gráfica de barras para visualizar las probabilidades de abandono de los clientes seleccionados con el modelo LR
        plt.figure(figsize=(20, 8))
        plt.bar(top_clientes_lr['Apellido'] + ' - ' + top_clientes_lr['ClienteId'].astype(str), top_clientes_lr['Probabilidad_Abandono_LR'], color='lightcoral')
        plt.xlabel('Apellidos de los Clientes')
        plt.ylabel('Probabilidad de Abandono (LR)')
        plt.title(f'Top {cant_top_clientes} Clientes con Mayor Probabilidad de Abandono (LR)')
        plt.xticks(rotation=90)  # Rotar los apellidos para una mejor visualización

        # Mostrar la gráfica utilizando st.pyplot
        st.pyplot(plt)
        
def entrenamientoArboles(cant_top_clientes):
    preprocesamiento()
    X_train_preprocessed, X_test_preprocessed, X_train, X_test, y_train, y_test = preprocesamiento()
    
    # Crear y entrenar el modelo seleccionado (Bosques Aleatorios)
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_preprocessed, y_train)

    # Predecir en el conjunto de prueba
    y_pred = rf_model.predict(X_test_preprocessed)

    # Calcular la precisión en el conjunto de prueba
    accuracy_rf = accuracy_score(y_test, y_pred)
    
    # Mostrar la precisión en la interfaz
    st.write(f'Precisión del modelo en el conjunto de prueba: {accuracy_rf:.2f}')

    # Obtener las probabilidades de abandono para cada cliente utilizando el modelo de Bosques Aleatorios
    y_probs_rf = rf_model.predict_proba(X_test_preprocessed)[:, 1]

    # Crear un DataFrame con los resultados
    clientes_probables_abandono_rf = pd.DataFrame({'Apellido': X_test['Apellido'], 'ClienteId': X_test['ClienteId'],'Probabilidad_Abandono_RF': y_probs_rf})

    # Ordenar el DataFrame por las probabilidades en orden descendente
    clientes_probables_abandono_rf = clientes_probables_abandono_rf.sort_values(by='Probabilidad_Abandono_RF', ascending=False)

    # Validar que la cantidad ingresada sea válida
    if cant_top_clientes <= 0:
        st.warning("La cantidad ingresada debe ser un número positivo.")
    else:
        # Tomar solo la cantidad especificada de clientes
        top_clientes_rf = clientes_probables_abandono_rf.head(cant_top_clientes)

        # Crear una gráfica de barras para visualizar las probabilidades de abandono de los clientes seleccionados con el modelo RF
        plt.figure(figsize=(20, 8))
        plt.bar(top_clientes_rf['Apellido'] + ' - ' + top_clientes_rf['ClienteId'].astype(str), top_clientes_rf['Probabilidad_Abandono_RF'], color='lightcoral')
        plt.xlabel('Apellidos de los Clientes')
        plt.ylabel('Probabilidad de Abandono (RF)')
        plt.title(f'Top {cant_top_clientes} Clientes con Mayor Probabilidad de Abandono (RF)')
        plt.xticks(rotation=90)  # Rotar los apellidos para una mejor visualización

        # Mostrar la gráfica utilizando st.pyplot
        st.pyplot(plt)

def entrenamientoSVM(cant_top_clientes):
    preprocesamiento()
    X_train_preprocessed, X_test_preprocessed, X_train, X_test, y_train, y_test = preprocesamiento()

    # Crear y entrenar el modelo SVM
    svm_model = SVC(probability=True)
    svm_model.fit(X_train_preprocessed, y_train)

    # Predecir en el conjunto de prueba
    y_pred = svm_model.predict(X_test_preprocessed)

    # Calcular la precisión en el conjunto de prueba
    accuracy_svm = accuracy_score(y_test, y_pred) * 100

    # Mostrar la precisión en la interfaz
    st.write(f'Precisión del modelo SVM en el conjunto de prueba: {accuracy_svm:.2f}%')

    # Obtener las probabilidades de abandono para cada cliente
    y_probs_svm = svm_model.predict_proba(X_test_preprocessed)[:, 1]

    # Crear un DataFrame con los resultados
    clientes_probables_abandono_svm = pd.DataFrame({'Apellido': X_test['Apellido'], 'ClienteId': X_test['ClienteId'],'Probabilidad_Abandono_SVM': y_probs_svm})

    # Ordenar el DataFrame por las probabilidades en orden descendente
    clientes_probables_abandono_svm = clientes_probables_abandono_svm.sort_values(by='Probabilidad_Abandono_SVM', ascending=False)

    # Validar la cantidad ingresada
    if cant_top_clientes <= 0:
        st.warning("La cantidad ingresada debe ser un número positivo.")
    else:
        # Tomar solo la cantidad especificada de clientes
        top_clientes_svm = clientes_probables_abandono_svm.head(cant_top_clientes)

        # Crear una gráfica de barras para visualizar las probabilidades de abandono de los clientes seleccionados con el modelo SVM
        plt.figure(figsize=(20, 8))
        plt.bar(top_clientes_svm['Apellido'] + ' - ' + top_clientes_svm['ClienteId'].astype(str), top_clientes_svm['Probabilidad_Abandono_SVM'], color='lightcoral')
        plt.xlabel('Apellidos de los Clientes')
        plt.ylabel('Probabilidad de Abandono (SVM)')
        plt.title(f'Top {cant_top_clientes} Clientes con Mayor Probabilidad de Abandono (SVM)')
        plt.xticks(rotation=90)  # Rotar los apellidos para una mejor visualización

        # Mostrar la gráfica utilizando st.pyplot
        st.pyplot(plt)

def entrenamientoGradientBoosting(cant_top_clientes):
    preprocesamiento()
    X_train_preprocessed, X_test_preprocessed, X_train, X_test, y_train, y_test = preprocesamiento()

    # Crear y entrenar el modelo de Gradient Boosting
    gb_model = GradientBoostingClassifier()
    gb_model.fit(X_train_preprocessed, y_train)

    # Predecir en el conjunto de prueba
    y_pred = gb_model.predict(X_test_preprocessed)

    # Calcular la precisión en el conjunto de prueba
    accuracy_gb = accuracy_score(y_test, y_pred) * 100

    # Mostrar la precisión en la interfaz
    st.write(f'Precisión del modelo Gradient Boosting en el conjunto de prueba: {accuracy_gb:.2f}%')

    # Obtener las probabilidades de abandono para cada cliente
    y_probs_gb = gb_model.predict_proba(X_test_preprocessed)[:, 1]

    # Crear un DataFrame con los resultados
    clientes_probables_abandono_gb = pd.DataFrame({'Apellido': X_test['Apellido'],'ClienteId': X_test['ClienteId'], 'Probabilidad_Abandono_GB': y_probs_gb})

    # Ordenar el DataFrame por las probabilidades en orden descendente
    clientes_probables_abandono_gb = clientes_probables_abandono_gb.sort_values(by='Probabilidad_Abandono_GB', ascending=False)

    # Validar la cantidad ingresada
    if cant_top_clientes <= 0:
        st.warning("La cantidad ingresada debe ser un número positivo.")
    else:
        # Tomar solo la cantidad especificada de clientes
        top_clientes_gb = clientes_probables_abandono_gb.head(cant_top_clientes)

        # Crear una gráfica de barras para visualizar las probabilidades de abandono de los clientes seleccionados con el modelo Gradient Boosting
        plt.figure(figsize=(20, 8))
        plt.bar(top_clientes_gb['Apellido'] + ' - ' + top_clientes_gb['ClienteId'].astype(str), top_clientes_gb['Probabilidad_Abandono_GB'], color='lightcoral')
        plt.xlabel('Apellidos de los Clientes')
        plt.ylabel('Probabilidad de Abandono (GB)')
        plt.title(f'Top {cant_top_clientes} Clientes con Mayor Probabilidad de Abandono (GB)')
        plt.xticks(rotation=90)  # Rotar los apellidos para una mejor visualización

        # Mostrar la gráfica utilizando st.pyplot
        st.pyplot(plt)

# Función para buscar un cliente por ClienteId
def buscar_cliente(cliente_id):
    cliente = data[data['ClienteId'] == cliente_id]
    if not cliente.empty:
        # Mostrar los datos del cliente
        st.subheader(f'Datos del Cliente - ClienteId: {cliente_id}')
        st.dataframe(cliente)
    else:
        st.warning(f'Cliente con ClienteId "{cliente_id}" no encontrado.')

# Función principal de la aplicación Streamlit
def main():
    # Mostrar un análisis exploratorio de datos con datos limpios
    if st.button('Mostrar Análisis Exploratorio de Datos'):
        limpieza()
        plt.clf()
        analisisExploratorio()

    # Mostrar una sugerencia de que modelo seleccionar
    st.write("Sugerencia de que modelo utilizar según su precisión")
    if st.button('Mostrar Sugerencia'):
        seleccionModelo()

    cant_top_clientes = st.slider("Inserta la cantidad top de clientes que deseas ver: ")
    
    st.write("Elige el modelo de entrenamiento que que deseas utilizar: ")
    if st.button('Regresión Logística'):
        entrenamientoRegresion(cant_top_clientes)
    if st.button('Arboles de Decision'):
        entrenamientoArboles(cant_top_clientes)
    if st.button('SVM'):
        entrenamientoSVM(cant_top_clientes)
    if st.button('GradientBoosting'):
        entrenamientoGradientBoosting(cant_top_clientes)
    # Interfaz de búsqueda por ID
    st.subheader('Búsqueda de Cliente por ID')
    cliente_id = st.text_input("Introduce el ID del Cliente")
    if st.button("Buscar"):
        buscar_cliente(int(cliente_id))

if __name__ == '__main__':
    main()