import numpy as np

def get_sample(sample_idx, idx_to_char):
    text = "".join(idx_to_char[idx] for idx in sample_idx)
    text = text[0].upper() + text[1:]
    return text

def softmax(x):
    e_x = np.exp(x-np.max(x))
    return e_x/e_x.sum(axis=0)

# Trick to smooth loss
def smooth(loss, curr_loss):
    return loss*0.999 + curr_loss*0.001

def get_initial_loss(vocab_size, seq_length):
    return -np.log(1.0/vocab_size)/seq_length

def initialize_parameters(n_x, n_a, n_y):
    Wax = np.random.randn(n_a, n_x)*0.01 # Input to hidden
    Waa = np.random.randn(n_a, n_a) * 0.01 # Hidden to hidden
    Wya = np.random.randn(n_y, n_a) * 0.01 # hidden to output
    b = np.zeros((n_a, 1))
    by = np.zeros((n_y, 1))
    parameters = {"Wax": Wax, "Waa":Waa, "Wya":Wya, "b":b, "by":by}
    return parameters

def rnn_step_forward(parameters, a_prev, x):
    Wax, Waa, Wya, b, by = parameters['Wax'], parameters['Waa'], parameters['Wya'], parameters['b'], parameters['by']
    a_next = np.tanh(np.dot(Wax,x)+np.dot(Waa,a_prev)+b)
    z = np.dot(Wya, a_next)
    y_hat = softmax(z)
    # probabilities for next chars
    return a_next, y_hat

def rnn_forward(X, Y, a0, parameters, vocab_size = 27):
    # Initialize x, a and y_hat as empty dictionarie
    x, a, y_hat = {}, {}, {}

    a[-1] = np.copy(a0)

    # initialize your loss to 0
    loss = 0

    for t in range(len(X)):

        # Set x[t] to be the one-hot vector representation of the t'th character in X.
        # if X[t] == None, we just have x[t]=0. This is used to set the input for the first timestep to the zero vector.
        x[t] = np.zeros((vocab_size, 1))
        if X[t] != None:
            x[t][X[t]] = 1

        # Run one step forward of the RNN
        a[t], y_hat[t] = rnn_step_forward(parameters, a[t-1], x[t])

        # Update the loss by substracting the cross-entropy term of this time-step from it.
        loss -= np.log(y_hat[t][Y[t], 0])

    cache = (y_hat, a, x)
    return loss, cache

def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    gradients['dWya'] += np.dot(dy, a.T)
    gradients["dby"] += dy
    da = gradients["da_next"] + np.dot(Wya.T, dy)
    daraw = (1-a*a)*da # backprop through tanh nonlinearity
    gradients["dWax"] += np.dot(daraw, x.T)
    gradients["dWaa"] += np.dot(daraw, a_prev.T)
    gradients["db"] += daraw
    gradients["da_next"] = np.dot(Waa.T, daraw)

    return gradients


def update_parameters(parameters, gradients, lr):

    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Wya'] += -lr * gradients['dWya']
    parameters['b']  += -lr * gradients['db']
    parameters['by']  += -lr * gradients['dby']
    return parameters


def rnn_backward(X, Y, parameters, cache):
    # Initialize gradients as an empty dictionary
    gradients = {}

    # Retrieve from cache and parameters
    (y_hat, a, x) = cache
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']

    # each one should be initialized to zeros of the same dimension as its corresponding parameter
    gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
    gradients['db'], gradients['dby'] = np.zeros_like(b), np.zeros_like(by)
    gradients['da_next'] = np.zeros_like(a[0])

    for t in reversed(range(len(X))):
        dy = np.copy(y_hat[t])
        dy[Y[t]] -= 1
        gradients = rnn_step_backward(dy, gradients, parameters, x[t], a[t], a[t - 1])

    return gradients, a



# parameters = initialize_parameters(27, 50, 27)
# X = [None] + [1,2,3,4]
# Y = [1,2,3,4] + [10]
# a_prev = np.zeros((50, 1))
# loss, cache = rnn_forward(X, Y, a_prev, parameters)
# rnn_backward(X, Y, parameters, cache)




