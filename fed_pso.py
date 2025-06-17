import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
# agregar
tf.config.run_functions_eagerly(True)

from keras.datasets import cifar10, mnist
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from keras.backend import image_data_format
from keras.applications.mobilenet import MobileNet
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
import numpy as np
import copy

from build_model import Model
import csv
import random
import time

# client config
NUMOFCLIENTS = 2 # number of client(as particles)
EPOCHS = 2 # number of total iteration
CLIENT_EPOCHS = 1 # number of each client's iteration
BATCH_SIZE = 64 # Size of batches to train on
ACC = 0.3 # 0.4
LOCAL_ACC = 0.7 # 0.6
GLOBAL_ACC = 1.4 # 1.0

DROP_RATE = 0 # 0 ~ 1.0 float value


# model config 
LOSS = 'categorical_crossentropy' # Loss function
NUMOFCLASSES = 10 # Number of classes
lr = 0.0025
# se elimino argumento lr = lr
#OPTIMIZER = SGD( momentum=0.9, decay=lr/(EPOCHS*CLIENT_EPOCHS), nesterov=False) # lr = 0.015, 67 ~ 69%


def write_csv(algorithm_name, data, lr):
    file_name = "results/{name}_drop{drop}_lr{lr}_clients{cli}_epcli{cli_epoch}_ep{epochs}_batch{batch}.csv"
    file_name = file_name.format(
        drop=DROP_RATE,
        name=algorithm_name,
        lr=lr,
        cli=NUMOFCLIENTS,
        cli_epoch=CLIENT_EPOCHS,
        epochs=EPOCHS,
        batch=BATCH_SIZE
    )

    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Loss", "Accuracy"])
        for i, (loss, acc) in enumerate(data):
            writer.writerow([i + 1, loss, acc])



def load_dataset():
    """
    This function loads the dataset provided by Keras and pre-processes it in a form that is good to use for learning.
    
    Return:
        (X_train, Y_train), (X_test, Y_test)
    """
    # Code for experimenting with CIFAR-10 datasets.
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    return (X_train, Y_train), (X_test, Y_test)



def init_model(train_data_shape):
    """
    Create a model for learning.

    Args:
        train_data_shape: Image shape, ex) CIFAR10 == (32, 32, 3)

    Returns:
        Sequential model
    """
    lr = 0.0025
    optimizer = SGD(learning_rate=lr, momentum=0.9, decay=lr/(EPOCHS*CLIENT_EPOCHS), nesterov=False)
    model = Model(loss=LOSS, optimizer=optimizer, classes=NUMOFCLASSES)
    return model.fl_paper_model(train_shape=train_data_shape)


def client_data_config(x_train, y_train):
    """
    Split the data set to each client. Split randomly selected data from the total dataset by the same size.
    
    Args:
        x_train: Image data to be used for learning
        y_train: Label data to be used for learning

    Returns:
        A dataset consisting of a list type of client sizes
    """
    client_data = [() for _ in range(NUMOFCLIENTS)] # () for _ in range(NUMOFCLIENTS)
    num_of_each_dataset = int(x_train.shape[0] / NUMOFCLIENTS)

    for i in range(NUMOFCLIENTS):
        split_data_index = []
        while len(split_data_index) < num_of_each_dataset:
            item = random.choice(range(x_train.shape[0]))
            if item not in split_data_index:
                split_data_index.append(item)
        
        new_x_train = np.asarray([x_train[k] for k in split_data_index])
        new_y_train = np.asarray([y_train[k] for k in split_data_index])

        client_data[i] = (new_x_train, new_y_train)

    return client_data

class particle():
    def __init__(self, particle_num, client, x_train, y_train):
        # for check particle id
        self.particle_id = particle_num
        
        # particle model init
        self.particle_model = client
        
        # best model init
        self.local_best_model = client
        self.global_best_model = client

        # best score init
        self.local_best_score = 0.0
        self.global_best_score = 0.0

        self.x = x_train
        self.y = y_train

        # acc = acceleration
        self.parm = {'acc':ACC, 'local_acc':LOCAL_ACC, 'global_acc':GLOBAL_ACC}
        
        # velocities init
        self.velocities = [None] * len(client.get_weights())
        for i, layer in enumerate(client.get_weights()):
            self.velocities[i] = np.random.rand(*layer.shape) / 5 - 0.10

    def train_particle(self):
        print("particle {}/{} fitting".format(self.particle_id+1, NUMOFCLIENTS))

        # set each epoch's weight
        step_model = self.particle_model
        step_weight = step_model.get_weights()
        
        # new_velocities = [None] * len(step_weight)
        new_weight = [None] * len(step_weight)
        local_rand, global_rand = random.random(), random.random()

        # PSO algorithm applied to weights
        for index, layer in enumerate(step_weight):
            new_v = self.parm['acc'] * self.velocities[index]
            new_v = new_v + self.parm['local_acc'] * local_rand * (self.local_best_model.get_weights()[index] - layer)
            new_v = new_v + self.parm['global_acc'] * global_rand * (self.global_best_model.get_weights()[index] - layer)
            self.velocities[index] = new_v
            new_weight[index] = step_weight[index] + self.velocities[index]

        step_model.set_weights(new_weight)
        # agregacion de .weights.h5
        save_model_path = 'checkpoint/checkpoint_particle_{}.weights.h5'.format(self.particle_id)
        mc = ModelCheckpoint(filepath=save_model_path, 
                            monitor='val_loss', 
                            mode='min',
                            save_best_only=True,
                            save_weights_only=True,
                            )
        hist = step_model.fit(x=self.x, y=self.y,
                epochs=CLIENT_EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=1,
                #bajar para pruebas
                validation_split=0.1,
                callbacks=[mc],
                )
        
        # train_score_acc = hist.history['accuracy'][-1]
        train_score_loss = hist.history['val_loss'][-1]

        step_model.load_weights(save_model_path)
        self.particle_model = step_model

        # if self.global_best_score <= train_score_acc:
        if self.global_best_score >= train_score_loss:
            self.local_best_model = step_model
            
        return self.particle_id, train_score_loss
    
    def update_global_model(self, global_best_model, global_best_score):
        if self.local_best_score < global_best_score:    
            self.global_best_model = global_best_model
            self.global_best_score = global_best_score

    def resp_best_model(self, gid):
        if self.particle_id == gid:
            return self.particle_model


def get_best_score_by_loss(step_result):
    # step_result = [[step_model, train_socre_acc],...]
    temp_score = 100000
    temp_index = 0

    for index, result in enumerate(step_result):
        if temp_score > result[1]:
            temp_score = result[1]
            temp_index = index

    return step_result[temp_index][0], step_result[temp_index][1]


def get_best_score_by_acc(step_result):
    # step_result = [[step_model, train_socre_acc],...]
    temp_score = 0
    temp_index = 0

    for index, result in enumerate(step_result):
        if temp_score < result[1]:
            temp_score = result[1]
            temp_index = index

    return step_result[temp_index][0], step_result[temp_index][1]


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_dataset()

    server_model = init_model(train_data_shape=x_train.shape[1:])
    print(server_model.summary())

    client_data = client_data_config(x_train, y_train)
    pso_model = []
    for i in range(NUMOFCLIENTS):
        pso_model.append(particle(particle_num=i, client=init_model(train_data_shape=x_train.shape[1:]), x_train=client_data[i][0], y_train=client_data[i][1]))

    server_evaluate_acc = []
    global_best_model = None
    global_best_score = 0.0

    for epoch in range(EPOCHS):
        server_result = []
        start = time.time()

        for client in pso_model:
            if epoch != 0:
                client.update_global_model(server_model, global_best_score)
            
            # local_model, train_score = client.train_particle()
            # server_result.append([local_model, train_score])
            pid, train_score = client.train_particle()
            rand = random.randint(0,99)

            # Randomly dropped data sent to the server
            drop_communication = range(DROP_RATE)
            if rand not in drop_communication:
                server_result.append([pid, train_score])
        
        # Send the optimal model to each client after the best score comparison
        gid, global_best_score = get_best_score_by_loss(server_result)
        for client in pso_model:
            if client.resp_best_model(gid) != None:
                global_best_model = client.resp_best_model(gid)

        server_model = global_best_model
        
        print("server {}/{} evaluate".format(epoch+1, EPOCHS))
        server_evaluate_acc.append(server_model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=1))

    write_csv("FedPSO", server_evaluate_acc, lr)
