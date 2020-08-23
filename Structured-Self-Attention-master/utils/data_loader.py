#Please create your own dataloader for new datasets of the following type

import torch
import pandas as pd
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split
from utils.train import *
 
def load_data_set(type,max_len,vocab_size,batch_size):
	"""
	Loads the dataset. Keras Imdb dataset for binary classifcation. Keras reuters dataset for multiclass classification

	Args:
	type   : {bool} 0 for binary classification returns imdb dataset. 1 for multiclass classfication return reuters set
	max_len: {int} timesteps used for padding
	vocab_size: {int} size of the vocabulary
	batch_size: batch_size
	Returns:
	train_loader: {torch.Dataloader} train dataloader
	x_test_pad  : padded tokenized test_data for cross validating
	y_test      : y_test
	word_to_id  : {dict} words mapped to indices


	"""
	INDEX_FROM=2
	if not bool(type):

		NUM_WORDS=vocab_size # only use top 1000 words
		dataset = pd.read_csv('df_raw_text2.csv')

		dataset = dataset[~dataset.TEXTOS.isnull()]
		dataset.drop_duplicates(subset ="DOCS_ID",keep = 'first', inplace = True)
		dataset = dataset[dataset.V_AMB != 's/d']
		dataset.drop_duplicates(subset ="TEXTOS",keep = 'first', inplace = True)

		texts1 = [' '.join(txt.splitlines()) for txt in dataset['TEXTOS']] # remove /n
		texts2 = [preprocess(txt) for txt in texts1] # preprocess. ie. remove punct., lowercase, etc.

		t = Tokenizer(
		num_words=10000,
		filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
		lower=True,
		split=" ",
		char_level=False,
		oov_token=1
		)

		t.fit_on_texts(texts2)
		x_test_seq = t.texts_to_sequences(texts2)

		word_to_id = {k:v for k,v in t.word_index.items()}
		word_to_id["<PAD>"] = 0
		word_to_id["<UNK>"] = 1
		id_to_word = {value:key for key,value in word_to_id.items()}

		y = [1 if elem == 'si' else 0 for elem in dataset.VIOLENCIA_DE_GENERO]

		x = np.array(x_test_seq)
		y = np.array(y)

		x_train, x_test, y_train, y_test = train_test_split(x,
		          y,
		          train_size=.75, 
		          random_state=1)


		x_train_pad = pad_sequences(x_train,maxlen=max_len)
		x_test_pad = pad_sequences(x_test,maxlen=max_len)


		train_data = data_utils.TensorDataset(torch.from_numpy(x_train_pad).type(torch.LongTensor),torch.from_numpy(y_train).type(torch.DoubleTensor))
		train_loader = data_utils.DataLoader(train_data,batch_size=batch_size,drop_last=True)
		return train_loader,x_test_pad,y_test,word_to_id

	else:
		from tensorflow.keras.datasets import reuters

		train_set,test_set = reuters.load_data(path="reuters.npz",num_words=vocab_size,skip_top=0,index_from=INDEX_FROM)
		x_train,y_train = train_set[0],train_set[1]
		x_test,y_test = test_set[0],test_set[1]
		word_to_id = reuters.get_word_index(path="reuters_word_index.json")
		word_to_id = {k:(v+3) for k,v in word_to_id.items()}
		word_to_id["<PAD>"] = 0
		word_to_id["<START>"] = 1
		word_to_id["<UNK>"] = 2
		word_to_id['<EOS>'] = 3
		id_to_word = {value:key for key,value in word_to_id.items()}
		x_train_pad = pad_sequences(x_train,maxlen=max_len)
		x_test_pad = pad_sequences(x_test,maxlen=max_len)


		train_data = data_utils.TensorDataset(torch.from_numpy(x_train_pad).type(torch.LongTensor),torch.from_numpy(y_train).type(torch.LongTensor))
		train_loader = data_utils.DataLoader(train_data,batch_size=batch_size,drop_last=True)
		return train_loader,train_set,test_set,x_test_pad,word_to_id
