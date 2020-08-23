# PRUEBA TECNICA - TEKAL
by Pedro Freitas

## Introduction

The task was a NLP document classification problem. We were also supposed to define which sentences were most helpful in classifying the document.

The first step I did was to use the Google API to [download and extract the text](https://github.com/ppfreitas/nlp_prueba/blob/master/1%20-%20Dataset.ipynb) from the documents. After that, I tried a few different approaches for the classification problem.

I started using a term-frequency representation of the corpus followed by a simple Logistic Regression. Next, I tried a Paragraph Vector representation (Doc2Vec) using the gensim package followed by a Logistic Regression.

Finally, I adapted an implementation of the A Structured Self-Attentive Sentence Embedding paper (https://arxiv.org/abs/1703.03130) by https://github.com/kaushalshetty/Structured-Self-Attention/ for the gender violence dataset. The main advantage of this model is the attention terms visualization output it generates.

Both the tf-idf and self attention models can help us understand what terms or sentences help the most in classifying each class. To do so in the tf-idf, I got the average vector of the term-frequency matrix for each class. Note that the value of each column of the matrix can be interpreted as the probability of that term being in the doc, given a classification. So comparing the difference between the average values for two different classes can give us a good idea on how the terms distribution over the different classes differ. With the self attention model we can also retrieve which sentences were most relevant in the self attention layer for the classification, providing a nice visualization when ploting a heatmap of the weigths over the text.

All the out-of-sample results can be seen [in this notebook](https://github.com/ppfreitas/nlp_prueba/blob/master/2%20-%20TFIDF%20and%20doc2vec.ipynb). To predict a new text, an [application](http://ec2-35-180-254-8.eu-west-3.compute.amazonaws.com:8501/) (in initial stages) has been deployed. There are a few sample documents in the "Sample docs" folder to try the app.

## Conclusions
Both the tf-idf and doc2vec proved to be good classifiers when taken into account the difficulty in identifying minority classes in highly imbalanced datasets. The dataset was also relatively small, and the models would probably benefit from having more data. Considering that False Negatives are more problematic than False Positives, the tf-idf performed better than the doc2vec model.  

The [self-attention visualization](https://github.com/ppfreitas/nlp_prueba/blob/master/attention_print.png) heatmap does make sense for a lot of attention terms, however not all of them are intuitive to me. This model is underused in this analysis and with more time it could be used to deepen our analysis on the problem.

## Next Steps

The following actions would be done if the development of the application were to be continued:

* OCR extraction of some pdf documents that were not correctly processed in this preliminary version. 
* Improve corpus preprocessing. There are still some weird terms on the preprocessed corpus.
* Test models with pre trained embeddings (doc2vec and the attention model).
* Develop more tools to leverage the self-attention capabilities.
* Ensemble of different models.

## Key files description

* set_de_datos_con_perspectivas...: Original dataset.
* Diccionario_Set_de_Datos.pdf: Variable dictionary.
* 1 - Dataset.ipynb: Notebook used to construct the corpus and extract text from the files (given google drive id)
* 2 - TFIDF and doc2vec.ipynb: Notebook used to train the tf-idf and doc2vec models.
* df_raw_text2.csv: Dataset after extracting texts from documents.
* clean_dataset.csv: Dataset used in the models with variables already transformed.
* app.py: Streamlit code app.
* attention.html: visualization of the higher weights in the self-attention model
* Google.py: Code by https://learndataanalysis.org/ to initiate Google API service.
* Structured-Self-Attention-master: folder with the files for the attention model. Source: https://github.com/kaushalshetty/Structured-Self-Attention/


