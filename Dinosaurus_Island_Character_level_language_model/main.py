import numpy as np
from utils import *


data = open("dinos.txt").read().lower()

chars = sorted(list(set(data)))
char_to_idx = dict((c, i) for i, c in enumerate(chars))
idx_to_char = dict((i,c) for i, c in enumerate(chars))
data_size, vocab_size = len(data), len(chars)

def clip(gradients, max):
    dWax, dWaa, dWya, db, dby = gradients["dWax"], gradients["dWaa"], gradients["dWya"], gradients["db"], gradients["dby"]
    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, -max, max, out=gradient)

    gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}
    return gradients

def sample(parameters, char_to_idx, seed):
    # Retrieve parameters and relevant shapes from "parameters" dictionary
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = Wya.shape[0]
    n_a = Waa.shape[0]

    # Initialize idx to -1
    idx = -1
    counter = 0
    idx_new_line = char_to_idx["\n"]

    # Create an empty list of indices. This is the list which will contain the list of indices of the characters to generate
    indices = []
    # Representing the first character (initializing the sequence generation)
    x = np.zeros((vocab_size, 1))
    # Initialize a_prev as zeros
    a_prev = np.zeros((n_a, 1))

    while idx != idx_new_line and counter != 50:

        a_next = np.tanh(np.dot(Waa, a_prev)+np.dot(Wax, x)+b)
        y_hat = softmax(np.dot(Wya, a_next) + by)

        np.random.seed(seed)
        idx = np.random.choice(vocab_size, p=y_hat.ravel())

        indices.append(idx)
        x = np.zeros((vocab_size, 1))
        x[idx] = 1

        counter +=1
        seed +=1
        a_prev = a_next

    if counter == 50:
        indices.append("\n")

    return indices

def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):

    # Forward propagate through time
    loss, cache = rnn_forward(X, Y, a_prev, parameters)

    # Backpropagate through time
    gradients, a = rnn_backward(X, Y, parameters, cache)

    # Clip your gradients between -5 (min) and 5 (max)
    clip(gradients, 5)

    # Update parameters
    parameters = update_parameters(parameters, gradients, learning_rate)

    return loss, gradients, a[len(X) - 1]

def model(data_x, idx_to_char, char_to_idx, num_iterations = 35000, n_a = 50, dino_names = 10, vocab_size = 27):
    # Retrieve n_x and n_y from vocab_size
    n_x, n_y = vocab_size, vocab_size
    # Initialize parameters
    parameters = initialize_parameters(n_x, n_a, n_y)
    # Initialize loss (this is required because we want to smooth our loss
    loss = get_initial_loss(vocab_size, dino_names)
    # Build list of all dinosaur names (training examples).
    examples = [x.strip() for x in data_x]
    # Shuffle list of all dinosaur names
    np.random.seed(0)
    np.random.shuffle(examples)
    a_prev = np.zeros((n_a, 1))

    # Optimization loop
    for i in range(num_iterations):
        idx = i % len(examples)

        idx_example = [char_to_idx[c] for c in examples[idx]]
        X = [None] + idx_example

        idx_newline = char_to_idx["\n"]
        Y = X[1:] + [idx_newline]

        # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
        # Choose a learning rate of 0.01
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate=0.01)

        # Use a latency trick to keep the loss smooth. It happens here to accelerate the training.
        loss = smooth(loss, curr_loss)

        if i % 2000 == 0:
            print('Iteration: %d, Loss: %f' % (i, loss) + '\n')

            seed = 0
            for j in range(dino_names):
                idx_name = sample(parameters, char_to_idx, seed)
                name = get_sample(idx_name, idx_to_char)
                print(name.replace('\n', ''))

                seed +=1


model(data.split("\n"), idx_to_char, char_to_idx, 35000)



