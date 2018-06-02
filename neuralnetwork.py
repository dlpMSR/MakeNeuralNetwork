import numpy
import scipy.special
import matplotlib.pyplot as plt

class newralNetwork:

    def __init__(self, input_nodes, hidden_nodes,
                 output_nodes, learning_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate
        
        self.w_ih = numpy.random.rand(self.hnodes, self.inodes) - 0.5
        self.w_ho = numpy.random.rand(self.onodes, self.hnodes) - 0.5

        self.activation_function = lambda x: scipy.special.expit(x)
        pass


    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T        
        
        hidden_inputs = numpy.dot(self.w_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.w_ho, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs

        hidden_errors = numpy.dot(self.w_ho.T, output_errors)

        self.w_ho += self.lr * numpy.dot(output_errors*final_outputs*(1.0-final_outputs),
                                         numpy.transpose(hidden_outputs))
        
        self.w_ih = self.lr * numpy.dot(hidden_errors*hidden_outputs*(1.0-hidden_outputs),
                                        numpy.transpose(inputs))
                                    

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.w_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.w_ho, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
        

def main():
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.3
    n = newralNetwork(input_nodes, hidden_nodes,
                      output_nodes, learning_rate)
    
    training_data_file = open('./mnist_train.csv','r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:])/255.0*0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

    test_data_file = open('./mnist_test.csv'. 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    scorecard = []

    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        print(correct_label, 'correct_label')
            

if __name__ == '__main__':
    main()
