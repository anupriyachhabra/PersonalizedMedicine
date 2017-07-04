# https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Merge
from keras.preprocessing.text import Tokenizer
import csv
import sys


csv.register_dialect('piper', delimiter='|', quoting=csv.QUOTE_NONE)
csv.field_size_limit(sys.maxsize)
max_words = 1000
batch_size = 64

def train_model():

    epochs = 20

    X_train = []
    y_train = []
    x_gene = []
#
    #,
    print('Loading data...')
    with open('training_text', 'r') as text_file, open('training_variants', 'r') as variants_file:
        next(text_file, None)
        next(variants_file, None)
        csv_f1 = csv.reader(text_file, dialect='piper')
        csv_f2 = csv.reader(variants_file)
        for text, vars in zip(csv_f1, csv_f2):
            X_train.append(text[2])
            y_train.append(vars[3])
            x_gene.append(vars[1] +" "+vars[2])

    print(np.unique(y_train))
    print(len(np.unique(y_train)))
    num_classes = len(np.unique(y_train)) +1
    print(num_classes, 'classes')
    print(len(X_train), 'train sequences')
    print('Vectorizing sequence data...')
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    tokenizer.fit_on_sequences(X_train)
    x_train = tokenizer.sequences_to_matrix(X_train, mode='tfidf')
    print('x_train shape:', x_train.shape)

    print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
    y_train = keras.utils.to_categorical(np.asarray(y_train), num_classes)
    print('y_train shape:', y_train.shape)
    print('Building model...')
    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, input_shape=(max_words,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    tokenizer2 = keras.preprocessing.text.Tokenizer(num_words=max_words)
    tokenizer2.fit_on_texts(x_gene)
    x_gene = tokenizer2.texts_to_matrix(x_gene)
    print("x_gene shape", x_gene.shape)

    gene_model = Sequential()
    gene_model.add(Dense(512, input_shape=(max_words,)))
    gene_model.add(Activation('relu'))
    gene_model.add(Dropout(0.5))
    gene_model.add(Dense(256, input_shape=(max_words,)))
    gene_model.add(Activation('relu'))
    gene_model.add(Dropout(0.2))

    final_model = Sequential()
    final_model.add(Merge([model, gene_model], mode = 'concat'))
    final_model.add(Dense(num_classes))
    final_model.add(Activation('softmax'))

    final_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = final_model.fit(x = [x_train, x_gene], y = y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.2, shuffle=True)

    print(history.history['loss'])
    print(history.history['val_loss'])
    return final_model

def predict(model) :
    """

    :type model: Sequential
    """
    X_train = []
    y_train = []
    x_gene = []

    print('Loading data...')
    with open('test_text', 'r') as text_file, open('test_variants', 'r') as variants_file:
        next(text_file, None)
        next(variants_file, None)
        csv_f1 = csv.reader(text_file, dialect='piper')
        csv_f2 = csv.reader(variants_file)
        for text, vars in zip(csv_f1, csv_f2):
            X_train.append(text[2])
            x_gene.append(vars[1] +" "+vars[2])

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    tokenizer.fit_on_sequences(X_train)
    x_train = tokenizer.sequences_to_matrix(X_train, mode='tfidf')

    tokenizer2 = keras.preprocessing.text.Tokenizer(num_words=max_words)
    tokenizer2.fit_on_texts(x_gene)
    x_gene = tokenizer2.texts_to_matrix(x_gene)

    print(x_train.shape)
    predictions = model.predict_classes(x= [x_train, x_gene], batch_size=64)
    print("PREDICTIONS")
    print(predictions.shape)
    predictions = predictions.tolist()
    print(predictions)
    print(predictions[0])

    with open('output', 'w') as csvfile:
        result_writer = csv.writer(csvfile)
        result_writer.writerow(['ID', 'class1', 'class2', 'class3', 'class4' , 'class5', 'class6', 'class7', 'class8', 'class9'])
        for i in range(len(predictions)):
            result_writer.writerow([i, int(predictions[i] - 1==0) , int(predictions[i] - 2==0), int(predictions[i] - 3==0),
                                    int(predictions[i] - 4==0), int(predictions[i] - 5==0) , int(predictions[i]-6 ==0),
                                    int(predictions[i]-7==0), int(predictions[i]-8 ==0), int(predictions[i]-9 ==0)])



if __name__ == '__main__':
    with open('output', 'w') as csvfile:
        result_writer = csv.writer(csvfile)
        result_writer.writerow(['ID', 'class1', 'class2', 'class3', 'class4' , 'class5', 'class6', 'class7', 'class8', 'class9'])
    model = train_model()
    predict(model)
