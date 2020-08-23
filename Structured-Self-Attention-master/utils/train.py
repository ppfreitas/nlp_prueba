import os
import io
import re
import spacy
import nltk.corpus
import textract
import numpy as np
import pandas as pd
import pickle
import multiprocessing

from utils.train import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import recall_score, plot_confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from gensim.models import doc2vec

cores = multiprocessing.cpu_count() # Count the number of cores in a computer

# preprocessing of raw texts
not_alphanumeric_or_space = re.compile(r'[^\w|\s]')
nlp = spacy.load('es_core_news_sm')


def preprocess(doc):
    doc = re.sub(not_alphanumeric_or_space, '', doc) # remove punctuation
    #doc = re.sub('[^\w|\s]', '', doc) 
    doc = re.sub(r'\s+', ' ', doc, flags=re.I) 
    doc = re.sub(r'\s+[a-zA-Z]\s+', ' ', doc) # remove all single characters
    return doc.lower() # return lower case

# same as preprocess, but with lemmatization
def preprocess_lemm(doc):
    doc = re.sub(not_alphanumeric_or_space, '', doc) # remove punctuation
    #doc = re.sub('[^\w|\s]', '', doc) 
    doc = re.sub(r'\s+', ' ', doc, flags=re.I) 
    doc = re.sub(r'\s+[a-zA-Z]\s+', ' ', doc) # remove all single characters
    words = [t.lemma_ for t in nlp(doc) if t.lemma_ != '-PRON-']
    return ' '.join(words).lower()

def filter_stop_words(doc):
    filtered_words = [word for word in doc if word not in nltk.corpus.stopwords.words('spanish')]
    return ' '.join(filtered_words)

# retrieve words that are most useful to differentiate from one class to another in a tfidf space 
def get_most_relevant_words(vectorizer, X_train_m, y_train, n):
    diff = X_train_m[y_train == 1].toarray().mean(axis=0)/X_train_m[y_train == 0].toarray().mean(axis=0)
    idxs = np.argpartition(diff, -n)[-n:]
    return np.array(vectorizer.get_feature_names())[idxs.astype(int)]


# function to perform grid search cross validation and return best vectorizer params
def train_tf_idf(X, y):
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(use_idf = True)),
        ('oversamp', SMOTE(sampling_strategy = 0.4)),
        ('clf', LogisticRegression(solver="lbfgs")),
    ])

    parameters = {
        'vect__max_df': (0.7, 0.8, 0.9),
        'vect__min_df': (0., 0.02, 0.05),
        'vect__ngram_range': ((1, 1), (1, 2)),# unigrams or bigrams
        'vect__max_features': (500, 1000, 1500, 2000),
    }

    grid_search = GridSearchCV(pipeline, parameters, scoring= 'recall',
                               n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_score_, grid_search.best_params_


# using vectorizer params defined by grid_search, train model with MATERIA concat as feature
def train_tf_idf_materia(X_train1,X_train2, X_test1, X_test2, y_train, y_test, params):

    # take vectorizer params from gridsearch_cv and define it
    max_df = params['vect__max_df']
    min_df = params['vect__min_df']
    ngram_range = params['vect__ngram_range']
    max_feature = params['vect__max_features']

    vectorizer = TfidfVectorizer(min_df = min_df, 
                         max_df = max_df, 
                         max_features=max_feature,
                         ngram_range = ngram_range,
                         use_idf=True)

    # train set transformations: tfidf, concat with MATERIA, oversampling
    X_train_m = vectorizer.fit_transform(X_train1)             
    X_train = pd.concat([pd.DataFrame(X_train_m.toarray()),X_train2.reset_index(drop=True)],axis=1,
                        join = 'inner',ignore_index=True)
    sm = SMOTE(sampling_strategy = 0.4)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

    # test set transformation: tfidf, concat with MATERIA
    X_test_m = vectorizer.transform(X_test1)
    X_test = pd.concat([pd.DataFrame(X_test_m.toarray()),X_test2.reset_index(drop=True)],axis=1,
                       join = 'inner',ignore_index=True)

    # classifier: CV train set and predict test set
    lrg = LogisticRegression(solver="lbfgs", class_weight='balanced')
    lrg.fit(X_train_sm, y_train_sm)
    y_hat = lrg.predict(X_test)

    return lrg, vectorizer, y_hat, y_test, X_test, X_train_m


def get_most_relevant_words(vect, X_train_m, y_train, n):
    diff = X_train_m[y_train == 1].toarray().mean(axis=0)/X_train_m[y_train == 0].toarray().mean(axis=0)
    idxs = np.argpartition(diff, -n)[-n:]
    return np.array(vect.get_feature_names())[idxs.astype(int)]


def grid_search_doc2vec(X_texto, X_materia, y_train):
    lr_acc = []
    sizes = [500, 1000, 1500] # size of the vector
    counts = [1,5] #min_count for words to be included in vocab
    dms = [0] # distributed memory’ (PV-DM) is used. Otherwise, distributed bag of words (PV-DBOW)
    epoch = 10
    
    docs_token = [doc.split() for doc in X_texto]
    documents = [doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(docs_token)]

    for size in sizes:
        for count in counts:
            for dm in dms:

                # define model, vocab and train it
                model = doc2vec.Doc2Vec(vector_size=size, min_count=count, epochs=epoch,
                                       workers = cores-1, dm= dm)
                model.build_vocab(documents)
                model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

                # get vector representation of docs
                X_vector = []
                for doc in docs_token:
                    vector = model.infer_vector(doc)
                    X_vector.append(vector)

                X = pd.concat([X_materia.reset_index(drop=True),pd.DataFrame(X_vector)], axis=1)
                
                #oversampling
                sm = SMOTE(sampling_strategy = 0.4)
                X_train_sm, y_train_sm = sm.fit_resample(X, y_train)
                
                # classifier: CV score
                lrg = LogisticRegression(solver="lbfgs", class_weight='balanced')
                scores = cross_val_score(lrg,X=X_train_sm,y=y_train_sm,cv=5,n_jobs=-1,verbose=1, scoring = 'recall')

                lr_acc.append((scores.mean(),size,count,dm))
                
                results = pd.DataFrame(lr_acc)
                
                parameters = {
                    'size': results.sort_values(0).iloc[-1,:][1],
                    'count': results.sort_values(0).iloc[-1,:][2],
                    'dm': results.sort_values(0).iloc[-1,:][3]
                }
            
    return results, parameters

def train_doc2vec_classifier(X1, X2, y, embedd):

    docs_token = [doc.split() for doc in X1]
    documents = [doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(docs_token)]  
    
    # get vector representation of docs
    X_vector = []
    for doc in docs_token:
        vector = embedd.infer_vector(doc)
        X_vector.append(vector)

    X = pd.concat([X2.reset_index(drop=True),pd.DataFrame(X_vector)], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        train_size=.75, 
                                                        random_state=1)
    
    #oversampling training set
    sm = SMOTE(sampling_strategy = 0.4)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    
    lr = LogisticRegression(solver="lbfgs", class_weight='balanced', max_iter=1000)
    lr.fit(X_train_sm, y_train_sm)
    y_hat = lr.predict(X_test)
    
    return lr, y_hat, X_test, y_test


def text_from_doc(name):
    with open('teste.pdf', "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(name.read())
    return textract.process('teste.pdf').decode('utf-8')


def predict_from_txt(text, materia = 'Otro'):

    text = [text]

    texts1 = [' '.join(txt.splitlines()) for txt in text] # remove /n
    texts2 = [preprocess_lemm(txt1) for txt1 in texts1] # preprocess. ie. remove punct., lowercase, etc.

    tfidf_vects = ('tfidf_genero_vect.sav', 'tfidf_fisica_vect.sav', 'tfidf_psic_vect.sav',
                   'tfidf_econ_vect.sav', 'tfidf_sex_vect.sav', 'tfidf_soc_vect.sav',
                   'tfidf_amb_vect.sav', 'tfidf_simb_vect.sav')

    tfidf_models = ('tfidf_genero_model.sav', 'tfidf_fisica_model.sav', 'tfidf_psic_model.sav',
                   'tfidf_econ_model.sav', 'tfidf_sex_model.sav', 'tfidf_soc_model.sav',
                   'tfidf_amb_model.sav', 'tfidf_simb_model.sav')

    doc2vec_models = ('doc2vec_genero_model.sav', 'doc2vec_fisica_model.sav', 'doc2vec_psic_model.sav',
                      'doc2vec_econ_model.sav', 'doc2vec_sex_model.sav', 'doc2vec_soc_model.sav',
                      'doc2vec_amb_model.sav', 'doc2vec_simb_model.sav')

    embedd = pickle.load(open('./models/dov2vec_embbed.sav', "rb"))

    if materia == 'Penal':
        X2 = (0,0,0)
    elif materia == 'Contravencional':
        X2 = (1,0,0)
    elif materia == 'Faltas':
        X2 = (0,1,0)
    else:
        X2 = (0,0,1)

    X1 = np.array(texts2)
    X2 = pd.DataFrame([X2])

    results_tf = []
    results_d2v = []

    for vect, tf_model, d2v_model in zip(tfidf_vects, tfidf_models, doc2vec_models):

        # load models
        vectorizer = pickle.load(open(os.path.join('./models', vect), "rb"))
        tfmodel = pickle.load(open(os.path.join('./models', tf_model), "rb"))
        d2vmodel = pickle.load(open(os.path.join('./models', d2v_model), "rb"))

        # tfidf transformations
        Xtf = vectorizer.transform(X1)
        Xtf = pd.concat([pd.DataFrame(Xtf.toarray()),X2], axis=1)

        # d2v transformations
        doc_token = [doc.split() for doc in X1]
        documents = [doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(doc_token)]  

        X_vector = []
        for doc in doc_token:
            vector = embedd.infer_vector(doc)
            X_vector.append(vector)

        Xd2v = pd.concat([X2.reset_index(drop=True),pd.DataFrame(X_vector)], axis=1)

        results_tf.append(tfmodel.predict(Xtf))
        results_d2v.append(d2vmodel.predict(Xd2v))

    prediction = pd.DataFrame([results_tf,results_d2v], columns= ['Violencia de genero', 'Fís', 'Psic',
                                                    'Econ','Sex','Soc','Amb','Simb'],index=['TF-IDF','Doc2Vec'])
    return prediction

def get_most_relevant_words(vect, X_train_m, y_train, n):
    diff = X_train_m[y_train == 1].toarray().mean(axis=0)/X_train_m[y_train == 0].toarray().mean(axis=0)
    idxs = np.argpartition(diff, -n)[-n:]
    return np.array(vect.get_feature_names())[idxs.astype(int)]

def get_most_relevant_words2(vect, X_train_m, dataset, variable = 'VIOLENCIA_DE_GENERO', n = 20):
    if variable == 'VIOLENCIA_DE_GENERO':
        y_train = np.array(dataset[variable])
    else:
        X_train_m = X_train_m[np.array(dataset['VIOLENCIA_DE_GENERO']) == 1]
        y_train = np.array(dataset[variable])
        y_train = y_train[np.array(dataset['VIOLENCIA_DE_GENERO']) == 1]
        
    diff = X_train_m[y_train == 1].toarray().mean(axis=0)/X_train_m[y_train == 0].toarray().mean(axis=0)
    idxs = np.argpartition(diff, -n)[-n:]
    return np.array(vect.get_feature_names())[idxs.astype(int)]

def terms_associated(variable = 'VIOLENCIA_DE_GENERO', n = 20):
    
    dataset = pd.read_csv('clean_dataset.csv')
    vect = pickle.load(open('./models/vect_most_relevant.sav', "rb"))
    
    X_m = vect.transform(dataset.TEXTOS)
    n = n
    variable = variable
    
    return get_most_relevant_words2(vect, X_m, dataset, variable, n)







