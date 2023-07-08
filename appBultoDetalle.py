import streamlit as st
import os.path 
import pathlib
import pandas as pd
import openpyxl

st.set_page_config(page_title='TSC - APLICACIONES WEB',page_icon='🤡',layout='wide')
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
  st.write(dataFila)

