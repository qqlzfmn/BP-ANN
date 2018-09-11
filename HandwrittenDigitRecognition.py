import numpy as np
# library for plotting arrays
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork

if __name__ == '__main__':
    # number of input, hidden and output nodes
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    # learning rate
    learning_rate = 0.1

    # create instance of neural network
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # load the mnist training data CSV file into a list
    with open("mnist_dataset/mnist_train_100.csv", 'r') as training_data_file:
        training_data_list = training_data_file.readlines()

    # # train the neural network

    # epochs is the number of times the training data set is used for training
    epochs = 5

    for e in range(epochs):
        # go through all records in the training data set
        for record in training_data_list:
            # split the record by the ',' commas
            all_values = record.split(',')
            # scale and shift the inputs
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # creade the target output values (all 0.01, except the desired label which is 0.99)
            targets = np.zeros(output_nodes) + 0.01
            # all_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
            pass
        pass

    # load the mnist test data CSV file into a list
    with open("mnist_dataset/mnist_test_10.csv", 'r') as test_data_file:
        test_data_list = test_data_file.readlines()

    # # test the neural network

    # scorecard for how well the network performs, initially empty
    scorecard = []

    # go through all the records in the test data set
    for record in test_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # correctanswer is first value
        correct_label = int(all_values[0])
        # scale and shift the inputs
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # query the network
        outputs = n.query(inputs)
        # the index of the highest value corresponds to the label
        label = np.argmax(outputs)
        # append correct or incorrect to list
        if label == correct_label:
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)
            pass
        pass

    # calculate the performance score, the fraction of correct answers
    scorecard_array = np.asarray(scorecard)
    print("performance = ", scorecard_array.sum() / scorecard_array.size)

    # got the first test record
    all_values = test_data_list[0].split(',')
    # print the label
    print(all_values[0])

    image_array = np.asfarray(all_values[1:]).reshape((28, 28))
    plt.imshow(image_array, cmap='Greys', interpolation='None')

    n.query((np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
