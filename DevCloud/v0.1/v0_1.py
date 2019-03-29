#Model v0.1 Actvations Relu, Dropouts, BN on base NVIDIA Model only Center images

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization
from utils import INPUT_SHAPE, batch_generator
import tensorflow as tf
import argparse
import os

np.random.seed(0)

args_name_lst=['TimeStamp','POS_X','POS_Y','POS_Z','Q_W','Q_X','Q_Y','Q_Z','Throttle','Steering','Brake','Gear','Handbrake','RPM','Speed','center','right','left']

def plt_dynamic(x, vy, ty, ax, colors=['b']):
    ax.plot(x, vy, 'b', label="Validation Loss")
    ax.plot(x, ty, 'r', label="Train Loss")
    plt.legend()
    plt.grid()
    fig.canvas.draw()
    

def load_data(args):
    data_df = pd.read_csv(os.path.join('../AAA/data/merge.csv'), names=args_name_lst)

    X = data_df[['center', 'left', 'right']].values
    y = data_df['Steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X[:10,:], y[:10], test_size=args.test_size, random_state=0)
    y_train = y_train.astype(np.float)
    y_valid = y_valid.astype(np.float)
    return X_train, X_valid, y_train, y_valid

def build_model(args):
    model = Sequential()
    model.add(Lambda(lambda x: x/255, input_shape=(144,256,3)))
    model.add(Conv2D(24, (5, 5), activation='relu', subsample=(2, 2), kernel_initializer='he_normal'))
    model.add(Dropout(0.75))
    model.add(Conv2D(36, (5, 5), activation='relu', subsample=(2, 2), kernel_initializer='he_normal'))
    #model.add(Dropout(0.75))
    model.add(Conv2D(48, (5, 5), activation='relu', subsample=(2, 2), kernel_initializer='he_normal'))
    model.add(Dropout(0.75))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'))
    #model.add(Dropout(0.75))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.75))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.75))
    model.add(Dense(10, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1, kernel_initializer='he_normal'))
    model.summary()

    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    checkpoint = ModelCheckpoint('weights_v0_0-{epoch:03d}-{val_loss:.4f}.hdf5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 save_weights_only=True, 
                                 mode='auto')

    
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    history=model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
                        args.samples_per_epoch,
                        args.nb_epoch,
                        max_q_size=1,
                        validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1)
    
    score = modelb.evaluate(X_valid, Y_valid, verbose=0) 
    print('Test score:', score[0]) 
    print('Test accuracy:', score[1])

    fig,ax = plt.subplots(1,1)
    ax.set_xlabel('epoch') ; ax.set_ylabel('MSE')

    # list of epoch numbers
    x = list(range(1,args.nb_epoch+1))
    vy = history.history['val_loss']
    ty = history.history['loss']
    plt_dynamic(x, vy, ty, ax)


def s2b(s):
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'

class args:

    data_dir='../AAA/images'
    test_size=0.2
    keep_prob=0.5
    nb_epoch=30
    samples_per_epoch=200
    batch_size=100
    save_best_only=False
    learning_rate=0.01
# print('-' * 30)
# print('Parameters')
# print('-' * 30)
# for key, value in args.items():
#     print('{:<20} := {}'.format(key, value))
# print('-' * 30)

data = load_data(args)
model = build_model(args)
train_model(model, args, *data)
