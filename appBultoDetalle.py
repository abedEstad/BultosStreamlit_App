import streamlit as st
import os.path 
import pathlib
import pandas as pd
import openpyxl

from sklearn import preprocessing 
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from yellowbrick.cluster import KElbowVisualizer

import gspread
from google.oauth2 import service_account
from google.oauth2.service_account import Credentials

st.set_page_config(page_title='***TSC - APLICACIONES WEB***',page_icon='🤡',layout='wide')
st.title(':sunglasses: :sun_with_face: :face_with_cowboy_hat: :green[Creación de Bultos por Machine Learning] :sunglasses: :sun_with_face: :face_with_cowboy_hat:')
st.write('_Esta es una version de app que permite subir un archivo excel, editarlo, guardarlo y exportarlo a tu directorio. EXCEL XLSX!_ :sunglasses:')

archivo_subida_excel= st.file_uploader('Subir Planilla',type='xlsm',accept_multiple_files=False, label_visibility="visible",help=None)

if archivo_subida_excel is not None:
  data = archivo_subida_excel.getvalue().decode('utf-8',errors='ignore')
  parent_path = pathlib.Path(__file__).parent.parent.resolve()
  save_path = os.path.join(parent_path, "data")
  nombre =os.path.join('',archivo_subida_excel.name)
  nombreFinal = nombre.split('.')[0]
  st.write('Nombre:',nombre)
  dfPlanilla = pd.read_excel(archivo_subida_excel,sheet_name='PLANILLA',engine='openpyxl',header=0,usecols="A:AP",na_filter=False)
  dfSelloCalidad = pd.read_excel(archivo_subida_excel,sheet_name='RESUMEN',engine='openpyxl',header=None,usecols="JIE:JIS",na_filter=False)
  dfPlanilla.rename(columns={ dfPlanilla.columns[0]: "DESP" }, inplace = True)
  dfPlanilla.rename(columns={ dfPlanilla.columns[36]: "PRODUCTO" }, inplace = True)
  dfPlanilla.rename(columns={ dfPlanilla.columns[13]: "FORMA" }, inplace = True)
  dfPlanilla.rename(columns={ dfPlanilla.columns[31]: "TOTAL" }, inplace = True)
  dfPlanillaFINAL=dfPlanilla[dfPlanilla["DESP"]!=""]
  dfPlanillaFINAL['DESP']=dfPlanillaFINAL['DESP'].astype('str')
  dfPlanillaFINAL['INFO']=dfPlanillaFINAL['INFO'].astype('str')
  dfPlanillaPREARMADOS= dfPlanillaFINAL[dfPlanillaFINAL['PRODUCTO'].isin(['PREARMADOA','PREARMADO'])]
  dfPlanillaPREARMADOS['DESP']=dfPlanillaPREARMADOS['DESP'].astype('str')
  dfPlanillaPREARMADOS['INFO']=dfPlanillaPREARMADOS['INFO'].astype('str')
  data_planilla_df=dfPlanillaFINAL[~dfPlanillaFINAL['PRODUCTO'].isin(['PREARMADOA','PREARMADO'])]
  data_planilla_df['DESP']=data_planilla_df['DESP'].astype('str')
  data_planilla_df['INFO']=data_planilla_df['INFO'].astype('str')
  ## reemplazamos los vacion por cero para el metodo de estandarizado
  data_planilla_df.iloc[:,13] = data_planilla_df.iloc[:,13].replace({'':0.00},regex=True) ## forma
  data_planilla_df.iloc[:,14] = data_planilla_df.iloc[:,14].replace({'':0.00},regex=True)
  data_planilla_df.iloc[:,15] = data_planilla_df.iloc[:,15].replace({'':0.00},regex=True)
  data_planilla_df.iloc[:,16] = data_planilla_df.iloc[:,16].replace({'':0.00},regex=True)
  data_planilla_df.iloc[:,17] = data_planilla_df.iloc[:,17].replace({'':0.00},regex=True)
  data_planilla_df.iloc[:,18] = data_planilla_df.iloc[:,18].replace({'':0.00},regex=True)
  data_planilla_df.iloc[:,19] = data_planilla_df.iloc[:,19].replace({'':0.00},regex=True)
  data_planilla_df.iloc[:,20] = data_planilla_df.iloc[:,20].replace({'':0.00},regex=True)
  data_planilla_df.iloc[:,21] = data_planilla_df.iloc[:,21].replace({'':0.00},regex=True)
  data_planilla_df.iloc[:,22] = data_planilla_df.iloc[:,22].replace({'':0.00},regex=True)
  data_planilla_df.iloc[:,23] = data_planilla_df.iloc[:,23].replace({'':0.00},regex=True)
  data_planilla_df.iloc[:,24] = data_planilla_df.iloc[:,24].replace({'':0.00},regex=True)
  data_planilla_df.iloc[:,25] = data_planilla_df.iloc[:,25].replace({'':0.00},regex=True)
  data_planilla_df.iloc[:,26] = data_planilla_df.iloc[:,26].replace({'':0.00},regex=True)
  data_planilla_df.iloc[:,27] = data_planilla_df.iloc[:,27].replace({'':0.00},regex=True)
  data_planilla_df.iloc[:,28] = data_planilla_df.iloc[:,28].replace({'':0.00},regex=True)
  data_planilla_df.iloc[:,29] = data_planilla_df.iloc[:,29].replace({'':0.00},regex=True)
  data_planilla_df.iloc[:,30] = data_planilla_df.iloc[:,30].replace({'':0.00},regex=True)
  data_planilla_df.iloc[:,31] = data_planilla_df.iloc[:,31].replace({'':0.00},regex=True)
  data_planilla_df.iloc[:,32] = data_planilla_df.iloc[:,32].replace({'':0.00},regex=True)
  data_planilla_df.iloc[:,33] = data_planilla_df.iloc[:,33].replace({'':0.00},regex=True)
  unicosLlaves = data_planilla_df.iloc[:,0].unique()
  len(unicosLlaves)
  lst = []
  lista = []

  dataFila=data_planilla_df.iloc[[1,2],13:34]
  dataFila['Cluster']=999

  for i in range(len(unicosLlaves)):
    dataPlanilla_df8=data_planilla_df[data_planilla_df["DESP"].isin([unicosLlaves[i]])]
    # data_planillaML8=dataPlanilla_df8.iloc[:,[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
    data_planillaML8=dataPlanilla_df8.iloc[:,13:34]
    min_max_scaler = preprocessing.MinMaxScaler() 
    escalado = min_max_scaler.fit_transform(data_planillaML8)
    df_escalado = pd.DataFrame(escalado) 
    #df_escalado = df_escalado.rename(columns = {0:'Forma', 1: 'a', 2: 'b',3:'c', 4:'d', 5:'Alfa', 6:'Beta',7:'Gama', 8: 'Radio', 9: 'CantxElemen',10:'Diam', 11:'Longitud', 12:'PesoDeFab',13:'Long_Fab', 14:'Cant_Total',15:'Total'})
    df_escalado = df_escalado.rename(columns = {0:'Forma',1:'Radio', 2: 'a', 3: 'b',4:'c', 5:'d', 6:'e', 7:'f', 8:'g', 9:'h', 10:'i', 11:'Alfa', 12:'Beta',
                                              13:'Gama',14:'CantxElemen', 15:'Cant_Total',16:'Diam', 17:'Longitud',18:'Total',19:'Long_Fab',20:'PesoDeFab'})

    #La data tiene 16 columnas
    if len(df_escalado) <= 21:
        data_planillaML8['Cluster']=99
        dataMenor21 = data_planillaML8
        dataFila = pd.concat([dataFila, dataMenor21], axis=0)
    else:
        model = KMeans()
        visualizer = KElbowVisualizer(model, k=(1,20),random_state = 12332)
        visualizer.fit(df_escalado)        
        #visualizer.show()
        cluster_codo =visualizer.elbow_value_
        cluster_codo
        lst.append(cluster_codo)
        #Modelo Jerarquico
        hc = AgglomerativeClustering(n_clusters = cluster_codo, 
                      affinity = 'euclidean', 
                      linkage = 'ward')
        etiquetas = hc.fit_predict(df_escalado)
        df_escalado['Cluster']=etiquetas
        data_planillaML8['Cluster']=etiquetas
        dataFinal = data_planillaML8
        #dataFinal = data_planillaML8.values.tolist()
        #lista.append(dataFinal)
        dataFila = pd.concat([dataFila, dataFinal], axis=0)


  ## fila 2 hacia adelante solo de la columna cluster
  dataFila_final=dataFila.iloc[2:,21]
  final =pd.concat([data_planilla_df,dataFila_final],axis=1)
  st.write(final99)

