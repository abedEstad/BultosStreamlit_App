import streamlit as st

st.set_page_config(page_title='***TSC - APLICACIONES WEB***',page_icon='🤡',layout='wide')
st.title(':sunglasses: :sun_with_face: :face_with_cowboy_hat: :green[Creación de Bultos por Machine Learning] :sunglasses: :sun_with_face: :face_with_cowboy_hat:')
st.write('_Esta es una version de app que permite subir un archivo excel, editarlo, guardarlo y exportarlo a tu directorio. EXCEL XLSX!_ :sunglasses:')

archivo_subida_excel= st.file_uploader('Subir Planilla',type='xlsm',accept_multiple_files=False, label_visibility="visible",help=None)

