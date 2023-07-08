import streamlit as st
import pandas as pd
import io
from io import BytesIO
import openpyxl

import os.path 
import pathlib

import gspread
import numpy as np
from pandas.core.frame import DataFrame
import pandas.io.formats.excel

from google.oauth2 import service_account
from google.oauth2.service_account import Credentials

from sklearn import preprocessing 
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from yellowbrick.cluster import KElbowVisualizer
import statistics

buffer = io.BytesIO()

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
  ##dataFila_final
  

  scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
  creds = Credentials.from_service_account_file('editor_sheet1.json',scopes=scope) 
  gc = gspread.authorize(creds)

  hojaCatalogoEmpaque = gc.open_by_key('13Mg1kfcd_lWeT-d-y5FMX9DnDuHCHAPLcHRPwgPbZGA').worksheet('Data')
  columna_referencia_CatalogoEmpaque = hojaCatalogoEmpaque.col_values(1)
  num_filas_CatalogoEmpaque = len(columna_referencia_CatalogoEmpaque)
  all_valores_CatalogoEmpaque = hojaCatalogoEmpaque.get_all_values()
  header_bddf_CatalogoEmpaque = all_valores_CatalogoEmpaque[4]
  valoresTrabajar_CatalogoEmpaque = all_valores_CatalogoEmpaque[5:num_filas_CatalogoEmpaque]
  basedatos_CatalogoEmpaque = pd.DataFrame(valoresTrabajar_CatalogoEmpaque) 
  basedatos_CatalogoEmpaque.columns = header_bddf_CatalogoEmpaque
  basedatos_CatalogoEmpaque.rename(columns={ basedatos_CatalogoEmpaque.columns[3]:'Bulto' }, inplace = True)


  basedatos_CatalogoEmpaque_U=basedatos_CatalogoEmpaque.iloc[:,[1,3]]
  basedatos_CatalogoEmpaque_U.iloc[:,0]=basedatos_CatalogoEmpaque_U.iloc[:,0].astype('int64')
  basedatos_CatalogoEmpaque_U.rename(columns={ basedatos_CatalogoEmpaque_U.columns[0]: "FORMA" }, inplace = True)
  final99 = final[final.loc[:,'Cluster']==99]

  if len(final99)>0:
    union = final99.merge(basedatos_CatalogoEmpaque_U[['Bulto','FORMA']],on='FORMA',how='left') #criterio es Bulto y NuevoGrupo es decir por cada bulto generamos un minibulto
    union=union.fillna('')
    union['Llave']=union['DESP']+'_'+union['Bulto']
    union['Llave']=union['Llave'].astype('str')
    tablaDinAOrden = union.sort_values(['Llave','TOTAL'])
    tablaDinAOrden['SumaPeso'] = tablaDinAOrden.groupby('Llave')[['TOTAL']].cumsum()
    pesoTope=250
    tablaDinAOrden['Agrupaciones'] = tablaDinAOrden.groupby('Llave')[['TOTAL']].cumsum() // pesoTope
    tablaDinAOrden['NuevoGrupo'] = (tablaDinAOrden.groupby('Llave')
                  .apply(lambda g: g.groupby('Agrupaciones').ngroup()+1)
                  .droplevel(0)
                )
    tablaDinAOrden.loc[:,'NuevoGrupo']=tablaDinAOrden.loc[:,'NuevoGrupo'].astype('str')
    tablaDinAOrden['IdGenereado']=tablaDinAOrden['Bulto']+"_"+tablaDinAOrden['NuevoGrupo']
    tablaDinAOrden['ClusterFinal'] = tablaDinAOrden.groupby("DESP")["IdGenereado"].transform(lambda x: pd.factorize(x)[0] + 1)
    
  finalNo99 = final[final.loc[:,'Cluster']!=99]
  finalNo99.loc[:,'Cluster'].unique()

  if len(finalNo99)>0:
    union2 = finalNo99.merge(basedatos_CatalogoEmpaque_U[['Bulto','FORMA']],on='FORMA',how='left') #criterio es Cluster y NuevoGrupo es decir por cada bulto generamos un minibulto
    union2=union2.fillna('')

    union2.loc[:,'Cluster']=union2.loc[:,'Cluster'].astype('str')
    union2['Llave']=union2.loc[:,'DESP']+'_'+union2.loc[:,'Cluster']
    finalNo99Orden = union2.sort_values(['Llave','TOTAL'])
    finalNo99Orden['SumaPeso'] = finalNo99Orden.groupby('Llave')[['TOTAL']].cumsum()
    pesoTope=250
    finalNo99Orden['Agrupaciones'] = finalNo99Orden.groupby('Llave')[['TOTAL']].cumsum() // pesoTope
    finalNo99Orden['NuevoGrupo'] = (finalNo99Orden.groupby('Llave')
                  .apply(lambda g: g.groupby('Agrupaciones').ngroup()+1)
                  .droplevel(0)
                )

    finalNo99Orden.loc[:,'NuevoGrupo']=finalNo99Orden.loc[:,'NuevoGrupo'].astype('str')
    finalNo99Orden['IdGenereado']=finalNo99Orden['Cluster']+"_"+finalNo99Orden['NuevoGrupo']
    finalNo99Orden['ClusterFinal'] = finalNo99Orden.groupby("DESP")["IdGenereado"].transform(lambda x: pd.factorize(x)[0] + 1)
  
  if len(final99)>0 & len(finalNo99)>0:
    dataConBultos = pd.concat([tablaDinAOrden, finalNo99Orden], axis=0)
  else:
    if  len(final99)>0 & len(finalNo99)==0:
      dataConBultos = tablaDinAOrden
    else:
      dataConBultos = finalNo99Orden
  
  if len(dfPlanillaPREARMADOS)>0:
    dataConBultosFINAL=pd.concat([dataConBultos, dfPlanillaPREARMADOS], axis=0)
    dataConBultosFINAL = dataConBultosFINAL.replace(np.nan, '', regex=True)
    dataConBultosFINAL = dataConBultosFINAL.replace("NaT", '', regex=True)
    dataConBultosFINAL = dataConBultosFINAL.replace("nan", '', regex=True)

  else:
    dataConBultosFINAL=dataConBultos
  
  dataConBultosFINAL=dataConBultosFINAL.sort_values(['DESP'],ascending=True)


  st.write(dataConBultosFINAL)  

  with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        pandas.io.formats.excel.ExcelFormatter.header_style =None
        # Write each dataframe to a different worksheet.
        dataConBultosFINAL.to_excel(writer, sheet_name='PLANILLA',index=False)
        workbook = writer.book
        worksheet = writer.sheets['PLANILLA']
        font_fmt_A = workbook.add_format({'font_name': 'Arial', 'font_size': 11, 'bold':True, 'font_color':'#FFFFFF' ,'bg_color':'#244062'})
        font_fmt_B = workbook.add_format({'font_name': 'Arial', 'font_size': 11, 'font_color':'#FFFFFF','bg_color':'#244062'})
        header_fmt = workbook.add_format({'font_name': 'Arial', 'font_size': 11, 'font_color':'black','bg_color':'#EBF1DE','text_wrap':True,
            'valign': 'center','align': 'center', 	'center_across':True})
        worksheet.set_row(0, None, header_fmt)  
        worksheet.set_column('A:A', None, font_fmt_A)
        worksheet.set_column('B:B', None, font_fmt_B)

        dfSelloCalidad.to_excel(writer, sheet_name='RESUMEN',index=False, header=None,startcol=7000)

        # Close the Pandas Excel writer and output the Excel file to the buffer
        #writer.save()
        writer.close()

        st.download_button(
        label='Descargar',
        data=buffer,
        file_name=nombreFinal+".xlsx",
        mime="application/vnd.ms-excel"
        )


