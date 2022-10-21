import random
import numpy as np
import mnist
import pickle


def sigmoid(z):
    """
    The sigmoid function.
     [30/10, 1]
    """
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


class Network:

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # [ch_out, 1]
        self.biases = [np.random.randn(ch_out, 1) for ch_out in sizes[1:]]
        # [ch_out, ch_in]
        self.weights = [np.random.randn(ch_out, ch_in)
                            for ch_in, ch_out in zip(sizes[:-1], sizes[1:])]
        self.params = dict()
        self.params['num_layers'] = self.num_layers
        self.params['sizes'] = self.sizes
        self.params['biases'] = self.biases
        self.params['weights'] = self.weights

    def forward(self, x):
        """

        :param x: [784, 1]
        :return: [30, 1]
        """

        for b, w in zip(self.biases, self.weights):
            # [30, 784]@[784, 1] + [30, 1]=> [30, 1]
            # [10, 30]@[30, 1] + [10, 1]=> [10, 1]
            x = sigmoid(np.dot(w, x)+b)
        return x

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None, save_path=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)

            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]

            # for every (x,y)
            for mini_batch in mini_batches:
                loss = self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j}:  Accuracy: {self.evaluate(test_data)/n_test*100}%, Loss: {loss}")
            else:
                print("Epoch {0} complete".format(j))
            if (j % (epochs // 10) == 0):
                if save_path:
                    self.params['num_layers'] = self.num_layers
                    self.params['sizes'] = self.sizes
                    self.params['biases'] = self.biases
                    self.params['weights'] = self.weights
                    self.save(save_path)
                    print(f"The params have been saved !    epoch:{j}")

    def Adam(self, training_data, epochs, mini_batch_size, eta, test_data=None, save_path=None):
        if test_data:
            n_test = len(test_data)

        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)

            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]

            # for every (x,y)
            for mini_batch in mini_batches:
                loss = self.update_mini_batch_adam(mini_batch, eta, j)
            if test_data:
                print(f"Epoch {j}:  Accuracy: {self.evaluate(test_data)/n_test*100}%, Loss: {loss}")
            else:
                print("Epoch {0} complete".format(j))
            if (j%(epochs//10)==0):
                if save_path:
                    self.params['num_layers'] = self.num_layers
                    self.params['sizes'] = self.sizes
                    self.params['biases'] = self.biases
                    self.params['weights'] = self.weights
                    self.save(save_path)
                    print(f"The params have been saved !    epoch:{j}")

    def update_mini_batch(self, mini_batch, eta):
        # https://en.wikipedia.org/wiki/Del
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        loss = 0

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w, loss_ = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            loss += loss_

        # tmp1 = [np.linalg.norm(b/len(mini_batch)) for b in nabla_b]
        # tmp2 = [np.linalg.norm(w/len(mini_batch)) for w in nabla_w]
        # print(tmp1)
        # print(tmp2)

        #weights = []
        #biases = []



        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        loss = loss / len(mini_batch)
        self.params = dict()
        return loss

    def update_mini_batch_adam(self, mini_batch, eta, i):
        # https://en.wikipedia.org/wiki/Del
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        loss = 0

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w, loss_ = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            loss += loss_

        # tmp1 = [np.linalg.norm(b/len(mini_batch)) for b in nabla_b]
        # tmp2 = [np.linalg.norm(w/len(mini_batch)) for w in nabla_w]
        # print(tmp1)
        # print(tmp2)

        weights = []
        biases = []
        # the declining rate for the moment estimation
        rho1, rho2 = 0.9, 0.999
        delta = 1e-8
        for w, nw in zip(self.weights, nabla_w):
            s = np.zeros_like(w)
            r = np.zeros_like(w)
            # update the first and the second moment
            s = rho1 * s + (1 - rho1) * nw
            r = rho2 * r + (1 - rho2) * np.square(nw)
            # get the partial first and second moment
            s_hat = s / (1 - rho1 ** (i + 1))
            r_hat = r / (1 - rho1 ** (i + 1))
            # use the partial first and second moment to modify the GD method
            w -= eta * s_hat / (len(mini_batch)*(np.sqrt(r_hat) + delta))
            weights.append(w)

        for b, nb in zip(self.biases, nabla_b):
            s = np.zeros_like(b)
            r = np.zeros_like(b)
            # update the first and the second moment
            s = rho1 * s + (1 - rho1) * nb
            r = rho2 * r + (1 - rho2) * np.square(nb)
            # get the partial first and second moment
            s_hat = s / (1 - rho1 ** (i + 1))
            r_hat = r / (1 - rho1 ** (i + 1))
            # use the partial first and second moment to modify the GD method
            b -= eta * s_hat / (len(mini_batch) * (np.sqrt(r_hat) + delta))
            biases.append(b)

        self.weights = weights
        self.biases = biases
        loss = loss / len(mini_batch)
        self.params = dict()
        return loss

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 1. forward
        activation = x
        # w*x = z => sigmoid => x/activation
        zs = [] # list to store all the z vectors, layer by layer
        activations = [x] # list to store all the activations, layer by layer
        for b, w in zip(self.biases, self.weights):
            # https://stackoverflow.com/questions/34142485/difference-between-numpy-dot-and-python-3-5-matrix-multiplication
            # np.dot vs np.matmul = @ vs element-wise *
            z = np.dot(w, activation)
            z = z + b # [256, 784] matmul [784] => [256]
            # [256] => [256, 1]
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        loss = np.power(activations[-1]-y, 2).sum()
        # 2. backward
        # (Ok-tk)*(1-Ok)*Ok
        # [10] - [10] * [10]
        delta = self.cost_prime(activations[-1], y) * sigmoid_prime(zs[-1]) # sigmoid(z)*(1-sigmoid(z))
        # O_j*Delta_k
        # [10]
        nabla_b[-1] = delta
        # deltaj * Oi
        # [10] @ [30, 1]^T => [10, 30]
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            # [30, 1]
            z = zs[-l]
            sp = sigmoid_prime(z)
            # sum()
            # [10, 30] => [30, 10] @ [10, 1] => [30, 1] * [30, 1]
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return nabla_b, nabla_w, loss

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.
        """
        test_results = [(np.argmax(self.forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_prime(self, output_activations, y):
        """
        Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations.
        """
        return output_activations-y

    def save(self, path):
        with open(path, 'wb') as f :
            pickle.dump(self.params, f)

    # 导入权重文件
    def load(self, path):
        with open(path, 'rb') as f :
            self.params = pickle.load(f)
            self.num_layers = self.params['num_layers']
            self.sizes = self.params['sizes']
            self.biases = self.params['biases']
            self.weights = self.params['weights']
        print("The params has been loaded !")


def main():

    x_train, y_train, x_test, y_test = mnist.load_data(reshape=[784, 1])
    print('x_train, y_train, x_test, y_test:', x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    np.random.seed(66)
    save_path = "Mnist_weight_bp.pickle"
    model = Network([784, 64, 10])
    load_weight = False
    if load_weight:
        model.load(save_path)
    data_train = list(zip(x_train, y_train))
    data_test = list(zip(x_test, y_test))
    model.SGD(training_data=data_train, epochs=10000000, mini_batch_size=128, eta=1e-4, test_data=data_test,
              save_path=save_path)


if __name__ == '__main__':
    main()