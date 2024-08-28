import streamlit as st
from main import ner_model

st.title("GliNer - NER Model")
st.write("Named Entity Recognition using gliner""")

text=st.text_input("Enter the Text", key="Text Box")
model_version=st.selectbox("Select a gliner model",["urchade/gliner_mediumv2.1"])

if st.button("run"):
    entities=ner_model(text, model_version=model_version)

    output={}

    for entity in entities:
        output[entity['text']]=entity['label']

    st.write(output)