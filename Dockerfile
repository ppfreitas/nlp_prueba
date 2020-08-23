FROM python:3.7

RUN pip install pandas numpy streamlit
RUN pip install scikit-learn gensim textract spacy nltk imbalanced-learn

RUN python -m spacy download es_core_news_sm

ADD . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]
