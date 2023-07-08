import streamlit as st
import os.path 
import pathlib

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

