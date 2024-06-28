import numpy
import matplotlib.pyplot as plt


data_file = open("dataset/mnist_test.csv")
data_list = data_file.readlines()
data_file.close()
print(len(data_list))
print(data_list[0])

#把数据依靠','区分，并分别读入
all_values = data_list[0].split(',')
#第一个值对应的是图片的表示的数字，所以我们读取图片数据时要去掉第一个数值
image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
onodes = 10
targets = numpy.zeros(onodes) + 0.01
targets[int(all_values[0])] = 0.99
print(targets)  #targets第8个元素的值是0.99，这表示图片对应的数字是7(数组是从编号0开始的).

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("dataset/mnist_train.csv")
trainning_data_list = training_data_file.readlines()
training_data_file.close()

for record in trainning_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    

scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为:",correct_number)
    #预处理数字图片
    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    #让网络判断图片对应的数字,推理
    outputs = n.query(inputs)
    #找到数值最大的神经元对应的 编号
    label = numpy.argmax(outputs)  
    print("output reslut is : ", label)
    #print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

scores_array = numpy.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)

epochs = 10

for e in range(epochs):
    for record in trainning_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
        #设置图片与数值的对应关系
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
