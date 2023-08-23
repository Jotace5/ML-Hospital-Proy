#Proyecto Hospital
#Analisis exploratorio de datos y Preparacion de los mismos para el modelo de Machine Learning.

#Importamos librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

 

#Cargamos la base de datos y la transformamos en un dataframe
df = pd.read_excel('Proyecto Integrador\Propuesta 1\BBDD_Hospitalización.xlsx')

#Exploramos el Dataframe
df.info() #Buscamos columnas vacias
df.describe() 

# Función para revisar el tipo de datos contenido dentro de cada columna

def verificar_tipo_datos(df):

    mi_dict = {"nombre_campo": [], "tipo_datos": [], "no_nulos_%": [], "nulos_%": []}

    for columna in df.columns:
        porcentaje_no_nulos = (df[columna].count() / len(df)) * 100
        mi_dict["nombre_campo"].append(columna)
        mi_dict["tipo_datos"].append(df[columna].apply(type).unique())
        mi_dict["no_nulos_%"].append(round(porcentaje_no_nulos, 2))
        mi_dict["nulos_%"].append(round(100-porcentaje_no_nulos, 2))

    df_info = pd.DataFrame(mi_dict)

    for columna in df.columns:
        print(columna, " (nulos) = ", df[columna].isnull().sum(),"/", len(df))

    return df_info

# Verificamos el tipo de datos de cada columna y valores nulos
verificar_tipo_datos(df)

#Procedemos a revisar los valores nulos de cada columna con valores NaN
df[df['PSA'].isna()] #Se deben tener en cuenta los 4 registros par el analisis.
df[df['BIOPSIAS PREVIAS'].isna()] #Debemos eliminar el registro 565, ya que no fue sometido a una biopsia, ni hospitalizado.
df[df['VOLUMEN PROSTATICO'].isna()] #Debemos eliminar el registro 565, ya que no fue sometido a una biopsia.
df[df['CUP'].isna()] #Debemos eliminar el registro 565, ya que no fue sometido a una biopsia.
df[df['ENF. CRONICA PULMONAR OBSTRUCTIVA'].isna()] #Consideramos los 2 registros, ya que presentan valores relevantes para el analisis y posterior entrenamiento del modelo.
df[df['AGENTE AISLADO'].isna()] #Consideramos a lo 15 registros de los 17 que salen como NaN en la columna AGENTE AISLADO. Procedemos a eliminar los registros 146 y 167 no fueron sometidos a una biopsia.
df[df['HOSPITALIZACION'].isna()] #Consideramos los 3 registros para el analisis y posterior entrenamiento del modelo.


#Revisamos la distribución estadística para el caso de las variables numéricas.
df.describe() 

#Notamos valores maximos atipicos para la variable EDAD.
#Buscamos que valores outliers existen en las variables numéricas.
plt.scatter(df['EDAD'], df['NUMERO DE MUESTRAS TOMADAS'])
plt.show() # Este primer grafico nos muestra que se tomaron muestras 2 personas mayores a 140 años.

# Buscamos en el dataframe los registros de edad mayores a 100 años.
df[df['EDAD'] > 100]
# Son 2 personas. Puede haber un error en la carga de datos. Consultamos con el cliente. (?)
# Procedemos a eliminar los registros, luego de la consulta.
df = df[df['EDAD'] < 100]
#Eliminamos los 2 registros con valores atipicos en la variable EDAD.
#Eliminamos el registro 161 y el 181 del dataframe.
df = df.drop([0, 1])

#Debemos eliminar el registro 565, ya que no fue sometido a una biopsia.
df = df.drop([565])

#Debemos eliminar el registro 146 y 167 no fueron sometidos a una biopsia.
df = df.drop([146, 167])



