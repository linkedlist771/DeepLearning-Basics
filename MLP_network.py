import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
import traceback

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.params = dict()
        self.params['num_layers'] = self.num_layers
        self.params['sizes'] = self.sizes
        self.params['biases'] = self.biases
        self.params['weights'] = self.weights

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None, save_path = None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j}, error: {self.evaluate(test_data)}" )
            else:
                print("Epoch {0} complete".format(j) )
            if (j%(epochs//10)==0):
                if save_path:
                    self.params['num_layers'] = self.num_layers
                    self.params['sizes'] = self.sizes
                    self.params['biases'] = self.biases
                    self.params['weights'] = self.weights
                    self.save(save_path)
                    print(f"The params have been saved !    epoch:{j}")

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        self.params = dict()

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)


    def evaluate(self, test_data):
        test_results = [(0.5*(self.feedforward(x))-y)**2
                        for (x, y) in test_data]
        return np.sum(test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

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


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))



def main():
    # Loading the MNIST data
    X = np.linspace(0, 2 * np.pi, 100).reshape((-1,1,1))
    Y = np.sin(X)
    save_path = "weight_bp.pickle"
    training_data = list(zip(X, Y))
    net = Network([1, 30, 1])
    train_net = True
    load_weight = True
    #net.load(save_path)
    if load_weight:
        try:
            net.load(save_path)  # 尝试读取权重
        except (Exception, BaseException) as e:
            print('{:*^60}'.format('直接打印出e, 输出错误具体原因'))
            print(e)
            print('{:*^60}'.format('使用repr打印出e, 带有错误类型'))
            print(repr(e))
            print('{:*^60}'.format('使用traceback的format_exc可以输出错误具体位置'))
            exstr = traceback.format_exc()
            print(exstr)

    if train_net:
        net.SGD(training_data, 1000000, 100, 0.001 , test_data=training_data, save_path=save_path)
    plt.figure()
    plt.scatter(np.array(X).flatten(), np.array(Y).flatten() , color="black")
    predict_y = [net.feedforward(x) for x in  X]
    #print(predict_y.shape)
    plt.plot(np.array(X).flatten(), np.array(predict_y).flatten() , color="red")
    plt.title("MLP fits sin by hands")
    plt.legend(["original", "predicted"])
    plt.savefig("手动实现_MLP.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    main()