# PRUEBA TECNICA - TEKAL
by Pedro Freitas

## Introduction

The problem to be solved was a NLP document classification problem. If possible, we were also supposed to define which sentences were most helpful in determining each class.

The first step I did was to use the Google API to download and extract the text from the documents. After that, I tried a few different approaches for the classification problem.

I started using a term-frequency representation of the corpus followed by a simple Logistic Regression. Next, I tried a Paragraph Vector representation (Doc2Vec) using the gensim package followed again by a Logistic Regression.

Finally, I adapted an implementation of the A Structured Self-Attentive Sentence Embedding paper (https://arxiv.org/abs/1703.03130) by https://github.com/kaushalshetty/Structured-Self-Attention/ for the gender violence dataset. The main advantage of this model is the nice attention visualization.

Both the tf-idf and self attention models can help us understand what terms or sentences help the most in classifying each class. To do so in the tf-idf, I got the average vector of the term-frequency matrix for each class. Note that the value of each column of the matrix can be interpreted as the probability of that term being in the doc, given a classification. So comparing the difference between the average value for two different classes can give us a good idea on how the terms distribution over the different classes differ. With the self attention model we can also retrieve which sentences were most relevant in the self attention layer for the classification, providing a nice visualization when ploting a heat map of the weigths over the text.

All the out-of-sample results can be seen in the  '2 - TFIDF and doc2vec.ipynb' notebook. To predict a new text, an application (in initial stage) has been deployed on 


## Key files description

* set_de_datos_con_perspectivas...: Original dataset.
* Diccionario_Set_de_Datos.pdf: Variable dictionary.
* 1 - Dataset.ipynb: Notebook used to construct the corpus and extract text from the files (given google drive id)
* 2 - TFIDF and doc2vec.ipynb: Notebook used to train the tf-idf and doc2vec models.
* df_raw_text2.csv: Dataset after extracting texts from documents.
* app.py: Streamlit app.

* client_secret.json: OAuth key for Google API.
* Google.py: Code by https://learndataanalysis.org/ to initiate Google API service.
* Structured-Self-Attention-master: folder with the files for the attention model. Source: https://github.com/kaushalshetty/Structured-Self-Attention/


## Next Steps

The following actions would be done if the development of the application were to be continued:

* Improve corpus preprocessing. There are still some weird terms on the preprocessed corpus.
* Test models with pre trained embeddings (doc2vec and the attention model).
* Develop more tools to leverage the self-attention capabilities.

