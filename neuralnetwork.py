import numpy
import scipy.special

class newralNetwork:
    def __init__(self, input_nodes, hidden_nodes,
                 output_nodes, learning_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate
        
        self.wih = numpy.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = numpy.random.rand(self.onodes, self.hnodes) - 0.5

        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T        
        
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        
        pass

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        pass


def main():
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3
    lerning_rate = 0.3
    n = newralNetwork(input_nodes, hidden_nodes,
                      output_nodes, learning_rate)
    pass


if __name__ == '__main__':
    main()
