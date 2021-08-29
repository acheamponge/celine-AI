import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from numpy import array

from tensorflow.keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")


folder = "./Data"
folders = os.listdir(folder)
#print(folders)

txt_files = []

for i in folders:
    for j in (os.listdir(folder+'/'+i)):
        txt_files.append('/'+i+'/'+j)

corpus = []
for i in txt_files:
    with open(folder+i,encoding='utf8') as f_input:
        corpus.append(f_input.read())

data = ""

for i in corpus:
    data+=i

def generate_seq(model, tokenizer, max_length, seed_text, n_words):
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # pre-pad sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
    return in_text

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
# create line-based sequences
sequences = list()
for line in data.split('\n'):
    encoded = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)
# pad input sequences
max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
# split into input and output elements
sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)


st.title('Céline: Générateur de texte AI')

st.write(
        f'<iframe width="560" height="315" src="https://www.youtube.com/embed/AzaTyxMduH4" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>',
        unsafe_allow_html=True,
    )
st.write('Ce projet est un générateur de texte utilisant les paroles des albums français de Céline Dion')

n = st.number_input('Tapez le nombre de mots que vous souhaitez générer', min_value=1, step=1 )


s = st.text_input('Tapez un mot ou des mots que vous souhaitez générer après')

if s and n:
    st.header((generate_seq(loaded_model, tokenizer, max_length-1, s, n)))

elif s and not n:
    st.write('Veuillez saisir des informations')

else:
    st.write('Veuillez saisir un mot et un nombre')
    
    
