import streamlit as st
import pandas as pd
import pickle
from utils import clean_text
from utils import vectorize
from transformers import AutoModelForSequenceClassification

st.subheader("Классификатор статей")

st.markdown("""
<style>
.big-font {
    font-size:20px !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown('<p class="big-font">Наш сервис позволяет узнать к какой категории относится научная публикация</p>', unsafe_allow_html=True)

input1 = st.text_area("Вставьте название публикации в поле ниже", key="title")
input2 = st.text_area("Вставьте abstract публикации в поле ниже", key="text")

def clear_text():
    st.session_state["title"] = ""
    st.session_state["text"] = ""

def get_class(input_text, model):
    vect_input = vectorize(pd.Series([clean_text(str(input_text))], name='text'))
    pred = model.predict(vect_input)
    codes = {0: 'cs', 1: 'econ', 2: 'eess', 3: 'math', 4: 'physics', 5: 'q-bio', 6: 'q-fin', 7: 'stat'}
    return codes[pred[0]]

co1, co2 = st.columns([1, 1])
with co1:
    if st.button("Проверить"):
        model = None
        try:
            model = pickle.load(open('../models/logreg.pkl', 'rb'))
        except:
            st.write('Ошибка загрузки модели')
        if model:
            try:
                ans = get_class(input2, model)
                strin = "Научная публикация относится к категории " + ans
            except:
                strin = "Что-то не так с публикацией"
            st.write(strin)

with co2:
    st.button("Очистить поля", on_click=clear_text)

st.write("Над проектом работали: Семен Семенов, Кирилл Кириллов, Елизавета Булгакова, Таисия Ускова")