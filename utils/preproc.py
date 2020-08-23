import os
import requests
import textract
import re
import io
from Google import Create_Service
from googleapiclient.http import MediaIoBaseDownload
import spacy


CLIENT_SECRET = 'client_secret.json'
API_NAME = 'drive'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/drive']

service = Create_Service(CLIENT_SECRET, API_NAME, API_VERSION, SCOPES)

# take google drive ids from shareable links taking into account different links templates
def get_id(url):
    if len(url.split('/')) > 6:
        doc_id = url.split('/')[-2]
    else:
        doc_id = url.split('=')[1]
    return doc_id


# extract texts from documents given doc_id
def get_text2(doc_id):

	# get metadata on doc given id
    file_name = service.files().get(fileId = doc_id).execute()
    file_path = os.path.join('./Docs', file_name['name'])

    # if google docs file, export method on API must be called
    if file_name['mimeType'] == 'application/vnd.google-apps.document':
        request = service.files().export(fileId= doc_id, mimeType = 'application/vnd.oasis.opendocument.text').execute()
        with open(file_path, 'wb') as f:
            f.write(request)
            text = textract.process(file_path, extension='odt')
            f.close()
    
    #if google drive, get method     
    else:
        request = service.files().get_media(fileId = doc_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fd=fh, request = request)
        done = False

        while not done:
            status, done = downloader.next_chunk()
            # print('Download progress {0}'.format(status.progress()*100))

        fh.seek(0)

        with open(file_path, 'wb') as f:
            f.write(fh.read())
            #extract text
            text = textract.process(file_path)
            f.close()

    return text

# preprocessing of raw texts
not_alphanumeric_or_space = re.compile(r'[^\w|\s]')
nlp = spacy.load('es_core_news_sm')

def preprocess(doc):
    doc = re.sub(not_alphanumeric_or_space, '', doc) # remove punctuation
    #doc = re.sub('[^\w|\s]', '', doc) 
    doc = re.sub(r'\s+', ' ', doc, flags=re.I) 
    doc = re.sub(r'\s+[a-zA-Z]\s+', ' ', doc) # remove all single characters
    return doc.lower() # return lower case

def preprocess_lemm(doc):
    doc = re.sub(not_alphanumeric_or_space, '', doc) # remove punctuation
    #doc = re.sub('[^\w|\s]', '', doc) 
    doc = re.sub(r'\s+', ' ', doc, flags=re.I) 
    doc = re.sub(r'\s+[a-zA-Z]\s+', ' ', doc) # remove all single characters
    words = [t.lemma_ for t in nlp(doc) if t.lemma_ != '-PRON-']
    return ' '.join(words).lower()


def get_most_relevant_words(vectorizer, X_train_m, y_train, n):
    diff = X_train_m[y_train == 1].toarray().mean(axis=0)/X_train_m[y_train == 0].toarray().mean(axis=0)
    idxs = np.argpartition(diff, -n)[-n:]
    return np.array(vectorizer.get_feature_names())[idxs.astype(int)]

