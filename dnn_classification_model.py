import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt


'Reading in both CSV files to process data.'
X_df = pd.read_csv('bank_transaction_features.csv')
Y_df = pd.read_csv('bank_transaction_labels.csv')

'Converting CSV to array type.'
X_dataset = X_df.values
Y_dataset = Y_df.values

'Dropping bank_transaction_id columns from both tables.'
X = X_dataset[:,1]
Y = Y_dataset[:,1:3]

'Separating labels for training and testing.'
Y_train = []    # 10,000 training samples
y_test = []     # 2,500 testing samples
for label, label_type in Y:
    if label_type == 'TRAIN':
        Y_train.append(label)
    else:
        y_test.append(label)

'Separating data for training and testing.'
X_train = X[0:len(Y_train)].astype(str)
x_test = X[len(Y_train):len(Y)].astype(str)

'Binarising Labels to feed into neural network'
'This is required as the network cannot process strings efficiently.'
lb = preprocessing.LabelBinarizer()
Y_train = lb.fit_transform(Y_train)
y_test = lb.fit_transform(y_test)

vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 10000

'Converting transaction descriptions into numeric data to'
'feed into the neural network.'
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)

training_sequences = tokenizer.texts_to_sequences(X_train)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(x_test)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

training_padded = np.array(training_padded)
training_labels = np.array(Y_train)
testing_padded = np.array(testing_padded)
testing_labels = np.array(y_test)

'Initialising the deep neural network structure.'
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='sigmoid')
])

'Running the deep neural network.'
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

num_epochs = 30
history = model.fit(training_padded, Y_train, epochs=num_epochs, verbose=1)

'Evaluating the deep neural network via test data set.'
print("\n")
results = model.evaluate(testing_padded, y_test)
print("test loss, test accuracy:", results)

'Evaluating the DNN using a confusion matrix.'
predictions = model.predict(testing_padded)
cm = multilabel_confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
print("\n")
print(cm)

'Evaluating the DNN using an ROC curve.'
fpr, tpr, thresholds = metrics.roc_curve(y_test.argmax(axis=1), predictions.argmax(axis=1), pos_label=2)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

