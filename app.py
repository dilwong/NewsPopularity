import streamlit as st

with st.form('articleForm', clear_on_submit=False):
    articleText = st.text_area('Article Text', '''
    Input the body of your news article here.
    ''')

    submitted = st.form_submit_button("Submit")

if submitted:
    st.write(articleText)