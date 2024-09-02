import streamlit as st
from main import ner_model
from datetime import datetime

st.title("GliNer - NER Model")
st.write("Named Entity Recognition using gliner""")

text=st.text_input("Enter the Text", key="Text Box")
model_version=st.selectbox("Select a gliner model",["urchade/gliner_mediumv2.1","urchade/gliner_small-v2.1"])

if st.button("run"):
    start_time=datetime.now()
    entities=ner_model(text, model_version=model_version)

    output={}

    for entity in entities:
        output[entity['label']]=entity['text']
    
    end_time=datetime.now()

    time_diff=(end_time-start_time).total_seconds()

    st.write(f"Processing time:{time_diff}s")
    st.write(output)