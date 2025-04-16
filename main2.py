import mnist_loader
training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()
import second_ml
net = second_ml.Network([784, 30, 10], cost=second_ml.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.5,
        lmbda = 5.0,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True,
        monitor_evaluation_cost=True,
        monitor_training_accuracy=True,
        monitor_training_cost=True)