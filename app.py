import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import pickle
import multiprocessing

from utils.train import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from gensim.models import doc2vec

cores = multiprocessing.cpu_count() # Count the number of cores in a computer

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title('Document classifier: gender violence')
st.write('Type the text in the area above or upload a file')

# receive text or upload file
doc_text = st.text_area('Text to analyze', 'Type something', height = 20)
uploaded_file = st.file_uploader("Choose a file. Working only with odt files.", type=['odt'])

if uploaded_file == None:
	text = doc_text
else:
	text = text_from_doc(uploaded_file)

text_pre1 = ' '.join(text.splitlines()) # remove /n
text_pre2 = preprocess(text_pre1) # preprocess. ie. remove punct., lowercase, etc.

if text_pre2 == ' ':
	st.warning('Could not get text. Please try a different pdf')

# select materia variable value
materia = st.radio(
	"What's the type of MATERIA?",
	('Penal', 'Contravencional', 'Faltas', 'Otro'))

# perform prediction
st.write(predict_from_txt(text, materia))
st.write('[1] means the document contains gender violence.')


# most relevant terms section
st.subheader('Terms most associated to each kind of violence:')
t_variable = st.selectbox('Select one type of gender violence:',
	('VIOLENCIA_DE_GENERO','V_FISICA','V_PSIC','V_ECON','V_SEX','V_SOC','V_AMB','V_SIMB'), index=0)
n = st.slider('Top n words associated to class', min_value=1, max_value=50, value=10, step=1)

st.table(terms_associated(variable = t_variable, n = n))


# print text from doc that is being classified
st.subheader('Document text:')
st.text(text)