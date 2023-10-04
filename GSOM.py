# Bibliotecas
import random
# import array
import time
import matplotlib.pyplot as plt
# import math
from math import exp
from math import sqrt
from math import log
import numpy as np
from sklearn.model_selection import train_test_split
from skmultilearn.dataset import load_dataset
import sklearn.metrics as metrics
import statistics as s
import multiprocessing



def fit(input, initial_width = 2, initial_height = 2, sf = 0.3, alfa = 0.01, epochs_growing=10, epochs_smoothing=5):
    """Fit's the data in the algoritm

    Parameters
    ----------
    input : numpy
        The training data
    initial_width : int
        The initial width of the map
    initial_height : int
        The initial height of the map
    sf : int
        The sf value
    alfa : int
        The initial learning rate of the algoritm
    epochs_growing : int
        The number of growing epochs
    epochs_smoothing : int
        The number of smooth epoches
    Returns
    --------
    None
    """

    init_grid(input, initial_width, initial_height, sf, alfa)
    start_growing_phase (input, epochs_growing)
    start_smoothing_phase (input, epochs_smoothing)

    return


def init_grid (input, initial_width = 2, initial_height = 2, sf = 0.3, alfa = 0.01):
    """Iniciates the GSOM algorithm

    Parameters
    ----------
    input : numpy
        The training data
    initial_width : int
        The initial width of the map
    initial_height : int
        The initial height of the map
    sf : int
        The sf value
    alfa: int
        The initial learning rate of the algoritm

    Returns
    --------
    None
    """

    samples_size = len(input[0])
    
    global weights
    weights = []
    
    global coordinate_map
    coordinate_map = []
    
    global acumulated_error
    acumulated_error = []  
    
    global gt
    gt = -samples_size * log(sf)            
    
    global fd
    fd = 0.5
    
    global initial_learning_rate
    initial_learning_rate = alfa
    
    total_neurons = initial_width * initial_height
    
    # create the weights vector
    for i in range (total_neurons):
        temp = []
        
        for j in range (samples_size):
            temp.append(random.random())
        
        weights.append(temp)
        
    # create the coordinate map
    for i in range (initial_width):
        for j in range (initial_height):
            coordinate_map.append([i, -1 * j])
            
    # create the acumulated error vector
    for i in range (total_neurons):
        acumulated_error.append(0)

    return
    
def start_growing_phase (input, epochs):
    """Iniciates the growing phase
    
    Parameters
    ----------
    input : numpy
        The training data
    epoches : int
        The epoches that the algoritm grow's

    Returns
    --------
    None
    """
    
    # print ("-----------------------")
    # print ("Growing Phase")
    # print ("-----------------------")
    
    # for each epoch
    for i in range (epochs):
        
        ts = time.time()
        
        # print ("Epoch: " + str(i))

        # show a random sample to the network
        # sample = random.choice(input)
        
        # show all the samples
        for j in range(len(input)):
        
            sample = input[j]
            
            # find the best matching unit
            bmu = find_best_matching_unit(sample)
            winner = bmu["winner"]
            error =  bmu["error"]

            # update the winner and the neighborhood
            update_neighborhood(winner, sample, i, epochs)
        
            # save the acumulated error vector
            acumulated_error[winner] = acumulated_error[winner] + error
            #print (acumulated_error)
        
            # check if the acumulated error is greater then the GT
            if (acumulated_error[winner] > gt): # & (len(weights) < 0.1*len(input)):
            
                boundary = check_boundary(winner, "available")
            
                if not boundary:
                    spread_error(winner)
                else:
                    for b in boundary:
                        grow (winner, b)
                        acumulated_error[winner] = 0
        # print("time:" + str(time.time() - ts) + " seconds")
    
    return
    
def start_smoothing_phase (input, epochs):
    """Iniciates the smooth phase
    
    Parameters
    ----------
    input : numpy
        The training data
    epoches : int
        The epoches that the algoritm smooth's

    Returns
    --------
    None
    """

    # print ("-----------------------")
    # print ("Smoothing Phase")
    # print ("-----------------------")
    
    # for each epoch
    for i in range (epochs):
        
        ts = time.time()
        
        # print ("Epoch: " + str(i))

        # show a random sample to the network
        #sample = random.choice(input)
        
        # show all the samples
        for j in range(len(input)):
        
            sample = input[j]
            
            # find the best matching unit
            bmu = find_best_matching_unit(sample)
            # print("O bmu dessa vez eh: ", bmu)
            winner = bmu["winner"]
            error =  bmu["error"]

            #Pode ser troacado por:
            # winner, error = find_best_matching_unit(sample).values()

            # update the winner and the neighborhood
            update_neighborhood(winner, sample, i, epochs)
            
        # print("time:" + str(time.time() - ts) + " seconds")
    
    return
    
def find_best_matching_unit (sample):
    """Find the neuron that beter corresponds to the current sample
    
    Parameters
    ----------
    sample : numpy
        The training data

    Returns
    --------
    result: tuple
        The winner neuron of the sample and the error of it
    None
    """
    
    result = dict()
    
    min_error = float('inf')
    winner = 0

    # for each neuron weight
    for i in range(len(weights)):
        
        # for each weight
        d = 0
        for j in range(len(weights[i])):
            
            # calculate the distance between the weight and the sample
            d = d + pow(weights[i][j] - sample[j], 2)
            
        d = sqrt(d)    
    
        
        # find the minimum error  
        if (d < min_error):
            min_error = d
            winner = i

    # print ("winner: " + str(winner))
    
    result["winner"] = winner
    result["error"] = min_error
    
    # return the winner
    return result
    
def update_neighborhood(winner, sample, iteration, epochs):
    """Updates the map
    """
    
    #print("updating neighborhood from neuron " + str(winner))
    
    # for each neuron weight
    for i in range(len(weights)):
        
        # get the distance between each neuron and the winner
        d = get_distance(i, winner)
        
        # for each weight of this neuron
        for j in range(len(weights[i])):
            
            # update the neuron weight
            weights[i][j] = weights[i][j] + neighbourhood_influence(d, iteration, epochs) * learning_rate(iteration, epochs) * (sample[j] - weights[i][j])
        
    return
    
def get_distance(n1, n2):
    """Get Euclidian distance from neurons
    """
    
    # calculate the distance between this neuron and the BMU
    x_n1, y_n1 = coordinate_map[n1]
    x_n2, y_n2 = coordinate_map[n2]

    # return (abs(x_n1 - x_n2) + abs(y_n1 - y_n2))
    a = x_n1 - x_n2
    b = y_n1 - y_n2

    return sqrt(a**2 + b**2)
    
def neighbourhood_influence (d, iteration, epochs):
    """Get the neighbourhood influence to update the neurons
    """
    
    starting_neighbourhood_influence = 0.6
    
    # sigma original:
    R = 3.8
    sigma = (starting_neighbourhood_influence * (1 - R)/ len(weights))

    # sigma used in Satelite project:
    # sigma = (starting_neighbourhood_influence * exp(-iteration / epochs))
    
    return (exp( -d / (2 * (pow(sigma, 2)) ))) 
    
def learning_rate(iteration, epochs):
    """Get the current learning rate to update the neurons
    """
    
    # original:
    R = 3.8
    return (initial_learning_rate * (1 - R)/ len(weights))

    # used in Satelite project:
    # return (initial_learning_rate * exp(-iteration / epochs))
    
def check_boundary (neuron, type = "available"):
    """Check's if there are available space in a given neuron neighborhood
    """
    
    if neuron is None:
        return []
    
    x, y = coordinate_map[neuron]
    
    all_boundary = ["L", "R", "B", "T"]
    boundary = []
    
    for i in range(len(coordinate_map)):
        
        # checking on left
        if (coordinate_map[i][0] == x - 1 and coordinate_map[i][1] == y):
            boundary.append("L")
            
        # checking on right
        if (coordinate_map[i][0] == x + 1 and coordinate_map[i][1] == y):
            boundary.append("R")
            
        # checking on bottom
        if (coordinate_map[i][0] == x and coordinate_map[i][1] == y - 1):
            boundary.append("B")
            
        # checking on top
        if (coordinate_map[i][0] == x and coordinate_map[i][1] == y + 1):
            boundary.append("T")

    if (type == "available"):
        return (list(set(all_boundary) - set(boundary)))
    elif (type == "unavailable"):
        return boundary
    
def grow (neuron, boundary):
    """ Creates a new neuron.
    """
    
    #print("growing on " + str(neuron) + " boundary: " + boundary)
    
    # append the weights
    add_weight(neuron, boundary)
    
    # append the coordinate map

    dict_neuron_grow = {
        "L": [coordinate_map[neuron][0]-1, coordinate_map[neuron][1]],
        "R": [coordinate_map[neuron][0]+1, coordinate_map[neuron][1]],
        "B": [coordinate_map[neuron][0], coordinate_map[neuron][1]-1],
        "T": [coordinate_map[neuron][0], coordinate_map[neuron][1]+1]
    }

    coordinate_map.append(dict_neuron_grow[boundary])
    
    # append the acumulated error vector
    acumulated_error.append(0)
    
    return
    
def add_weight(neuron, boundary):
    
    x, y = coordinate_map[neuron]

    # get the first neighbours
    first_neighbours = check_boundary(neuron, "unavailable")
    
    oposite = {"L": "R", "R": "L", "T": "B", "B": "T"}
    dict_neuron2 = {
        "L": get_neuron_by_coordinate_map (x-1, y),
        "R": get_neuron_by_coordinate_map (x+1, y),
        "B": get_neuron_by_coordinate_map (x, y-1),
        "T": get_neuron_by_coordinate_map (x, y+1)
    }

    if not(oposite[boundary] in first_neighbours):
        neuron2 = dict_neuron2[first_neighbours[0]]     # Case A: No consecutive neighbours, get the first one
    else:
        neuron2 = dict_neuron2[oposite[boundary]]  # Case B: There are two consecutive neighbours

            
    ##### --------------------------- #####
    ##### Calculating the new weights #####
    
    new_weights = []

    for i in range (len(weights[0])):
        # new_weights.append(weights[neuron][i] + abs(weights[neuron][i] - weights[neuron2][i]))
        new_weights.append(weights[neuron][i] + sqrt(weights[neuron][i]**2 + weights[neuron2][i]**2))

    weights.append(new_weights)
    
    return
    
def get_neuron_by_coordinate_map (x, y):
    """ Get neuron by the coordinate map
    """
    
    for i in range(len(coordinate_map)):
        if (coordinate_map[i][0] == x and coordinate_map[i][1] == y):
            return i
    
    return None

def spread_error (neuron):
    """ Calculates the spread error
    """
    
    x = coordinate_map[neuron][0]
    y = coordinate_map[neuron][1]
    
    for i in range(len(coordinate_map)):

        x_temp, y_temp = coordinate_map[i]
        
        # spreading to left
        if (x_temp == x-1 and y_temp == y):
            acumulated_error[i] += fd * acumulated_error[i]
        
        # spreading to right
        if (x_temp == x+1 and y_temp == y):
            acumulated_error[i] += fd * acumulated_error[i]
                    
        # spreading to bottom
        if (x_temp == x and y_temp == y-1):
            acumulated_error[i] += fd * acumulated_error[i]
                    
        # spreading to top
        if (x_temp == x and y_temp == y+1):
            acumulated_error[i] += fd * acumulated_error[i]

    # decrease the error of the winnerSS
    acumulated_error[neuron] = gt/2
    
    return
    

def get_neuron_labels (input, input_labels):
    """Get's the labels of a neuron. Only work to single-label problemns.
    
    Parameters
    ----------
    input : numpy
        The training data (X_train)
    input_labels : numpy
        The label of training data (y_train)

    Returns
    --------
    neuron_labels : dict
        The corespondent label to each neuron
    """
    
    neuron_labels = dict()
    frequency = dict()

    # extract the unique labels, and create a dict
    labels_set = set(input_labels)
    labels_unique = (list(labels_set))
    
    # create the frequency vector
    for i in range(len(weights)):
        frequency[i] = { i : j for j in input_labels[0] for i in labels_unique }

    # count the frequency of each label on each neuron
    for i in range(len(input)):
        
        bmu = find_best_matching_unit(input[i])
        n = bmu['winner']
        # frequency[bmu["winner"]][input_labels[i]] += 1
        frequency[n][input_labels]
    
    # set the most frequent label as the neuron label
    for i in range(len(weights)):
        
        count = 0
        top_label = ""
        
        for label in frequency[i]:
            
            if frequency[i][label] > count:
                count = frequency[i][label]
                top_label = label
        
        neuron_labels[i] = top_label
    
    return neuron_labels
    
def check_neuron_accuracy (input, neuron_labels, input_labels):
    """Check's neuron accuracy. Only works to single-lable problemns
    
    Parameters
    ----------
    input : numpy
        The training data (X_train)
    neuron_labels : dict
        The corespondent label to each neuron
    input_labels : numpy
        The label to predict (y_test)

    Returns
    --------
    None
    """
    
    neuron_stats = []
    
    # initialize neuron stats
    # [0] = hit
    # [1] = miss
    for i in range(len(weights)):
        neuron_stats.append([0, 0])
        
    # check each input classification, against neuron label
    for i in range(len(input)):
        
        bmu = find_best_matching_unit(input[i])
        n = bmu["winner"]
        
        if input_labels[i] == neuron_labels[n]:
            neuron_stats[n][0] += 1
        else:
            neuron_stats[n][1] += 1
    
    # printing neuron stats
    for i in range(len(neuron_stats)):
        
        sum = neuron_stats[i][0] + neuron_stats[i][1]

        if (sum == 0):
            print ("neuron [" + str(i) + "]: empty")
        else:
            print ("neuron [" + str(i) + "] [" + str(neuron_labels[i]) + "]:" + str(100 * neuron_stats[i][0]/sum) + "%")
    
    return
    
def generate_confusion_matrix(input, neuron_labels, input_labels):
    """Generates the confusion matrix. Only works to single-lable problemns
    
    Parameters
    ----------
    input : numpy
        The training data (X_train)
    neuron_labels : dict
        The corespondent label to each neuron
    input_labels : numpy
        The label to predict (y_test)

    Returns
    --------
    None
    """
    
    frequency = dict()

    # extract the unique labels, and create a dict
    labels_set = set(input_labels)
    labels_unique = (list(labels_set))
    
    print ("------")

    # create the label hit/miss vector
    label_hit = { i : 0 for i in labels_unique }
    label_miss = { i : 0 for i in labels_unique }
    
    total_hit = 0
    total_miss = 0
    
    # check each classification to generate the confusion matrix
    for i in range(len(input)):
        
        bmu = find_best_matching_unit(input[i])
        n = bmu["winner"]
        
        if input_labels[i] == neuron_labels[n]:
            label_hit[input_labels[i]] += 1
            total_hit += 1
        else:
            label_miss[input_labels[i]] += 1
            total_miss += 1
    
    for l in labels_unique:
        print ("class [" + str(l) + "] : " + str(100 * label_hit[l]/(label_hit[l] + label_miss[l])) + "%")
        
    print ("Total Accuracy : " + str(100 * total_hit/(total_hit + total_miss)) + "%")

    return

def get_neuron_labels_mutilabel (input, input_labels, factor=0.5, include=False):
    '''    
    Return predicted labels to each neuron.

    Parameters:
    --------
        input : numpy
            The data used to create the network (X_train)
        input_labels : numpy 
            The labels of the input data (y_train)
        factor : float
            The thresold used
        include : boolean
            If True, it will return the result without considering the thresold
    Returns
    --------
        label_weigths: dict
            The label result to each neuron
    '''
    
    neuron_labels = dict()
    frequency = dict()
    neuron_times_selected = list()

    # cria um vetor de frequências
    for i in range(len(weights)):
        frequency[i] = { i : 0 for i in range(len(input_labels[0])) }
        neuron_times_selected.append(0)
    
    # conta a frequência, sendo que cada vez que a label de um neurônio é 1, a frequência neste neurônio é acrescentada
    for i in range(len(input)):
        
        bmu = find_best_matching_unit(input[i])
        n = bmu['winner']
        
        # adding a point to every frequency
        for j in range(len(input_labels[i])):
            if input_labels[i][j] == 1: frequency[bmu["winner"]][j] += 1

        neuron_times_selected[bmu["winner"]] += 1
    
    label_weigths = dict()
    aux = dict()


    # Threshold
    for i in range(len(frequency)):
        if neuron_times_selected[i] == 0: neuron_times_selected[i] += 1     # turn's 0/0 into 0/1.

        for j in range(len(frequency[0])):
            frequency[i][j] = frequency[i][j]/neuron_times_selected[i]
            if (include):
                if (frequency[i][j] >= factor):
                    frequency[i][j] = 1
                else:
                    frequency[i][j] = 0
        label_weigths[i] = frequency[i]

    return label_weigths

def get_neuron_labels_mutilabel_list(input, labels_as_list, factor=0.5, include=False):
    '''    
    Return predicted labels to each neuron.

    Parameters:
    --------
        input : numpy
            The data used to create the network (X_train)
        input_labels : numpy 
            The labels of the input data (y_train)
        factor : float
            The thresold used
        include : boolean
            If True, it will return the result without considering the thresold
    Returns
    --------
        label_weigths: list
            The label result to each neuron
    '''
    r = get_neuron_labels_mutilabel(input, labels_as_list, factor)
    r_list = list()
    for i in range(len(r)):
        aux = []
        for j in range(len(r[0])):
            aux.append(r[i][j])
        r_list.append(aux)
    
    return r_list
    
def get_labels(input, predicted_labels):          # pega um conjunto de dados e diz sua classe, X_test, y_predidas_pelo_algoritmo
    """
    Return predicted labels to input data.

    Parameters:
    --------
        input : numpy
            Data to be labeled
        predicted_labels: 
            The label to each neuron
    
    Returns:
    --------
        labels: list
            Predicted label to input data
    """
    
    
    labels = list()

    for item in input:
        bmu = find_best_matching_unit(item)
        n = bmu['winner']

        labels.append(predicted_labels[n])

    return labels

def get_winner_neurons(input):    # diz os neurônios vencadores para cada dado de entrada
    """
    Return the winner neurons for each data.

    Parameters:
    --------
        input : numpy
            Data to be labeled
        predicted_labels: 
            The label to each neuron
    
    Returns:
    --------
        winner_neurons : list
            Returns the winner neuron to each sample of data input
    """

    winner_neurons = list()

    for item in input:
        bmu = find_best_matching_unit(item)
        n = bmu['winner']

        winner_neurons.append(n)

    return winner_neurons

def grade_density(winner_neurons, show=False):
    """
    Return the number of samples mapped to each neuron

    Parameters:
    --------
        winner_neurons : list
            The neurons used to label the data
        show : boolean
            Print the density to each neuron
    
    Returns:
    --------
        density : dict
            Returns the density to each neuron
    """
    density = dict()
    aux = list()

    for neuron in range(len(weights)):
        for index, item in enumerate(winner_neurons):
            if neuron == item:
                aux.append(index)
        density[neuron] = aux
        aux = []

    if (show == True):
        for i in range(len(density)):
            print("Neuron", i, ":", len(density[i]))

    return density

def performace_measures(y_real, y_predicted, show=False):
    """
    Return the metrics to valuate the algorims performace.

    Parameters:
    --------
        y_real : numpy
            The true labels
        y_predicted : numpy
            The predicted labels
        show : boolean
            If true, print the metrics result
    
    Returns:
    --------
        accuracy: float
            Returns the accuracy of the model
        precision: float
            Returns the precision of the model
        recall: float
            Returns the recall of the model
        f1_score: float
            Returns the f1_score of the model
        hamming_loss: float
            Returns the Hamming Loss of the model
    """
    VP = 0
    VN = 0
    FP = 0
    FN = 0

    for i in range(len(y_real)):
        for j in range(len(y_real[0])):
            if y_real[i][j] == 1:
                if y_predicted[i][j] == 1: 
                    VP += 1 
                else: 
                    FN += 1
            else:
                if y_predicted[i][j] == 1: 
                    FP += 1
                else: 
                    VN += 1           
    
    accuracy = (VP+VN)/(VP+VN+FP+FN)
    precision = VP/(VP+FP) if (VP+FP) != 0 else 0
    recall = VP/(VP+FN) if (VP+FN) != 0 else 0
    f1_score = 2 * precision * recall/(precision+recall) if (precision+recall) != 0 else 0
    hamming_loss = (FP+FN) /(len(y_real)*len(y_real[0]))

    if (show):
        print("Accuracy: {0:.2%}".format(accuracy))
        print("Precision: {0:.2%}".format(precision)) 
        print("Recall: {0:.2%}".format(recall)) 
        print("F1-Score: {0:.2%}".format(f1_score)) 
        print("Hamming Loss: {0:.2%}".format(hamming_loss)) 

    return (accuracy, precision, recall, f1_score, hamming_loss)


def selected_neurons_qdd(density):
    """
    Return the percentage of neurons used to label the data
    Parameters:
    --------
        density : dict
            The number of examples mapped to each neuron
    
    Returns:
    --------
        percentage : flot
            The percenge of neurons used to label the data 

    """
    value = 0
    for i in density.values():
        if (i != []):
            value += 1
    
    return value/len(density)


'''
======================
Automation to find the best metrics to each data.
======================
'''

from sklearn.preprocessing import MinMaxScaler
from skmultilearn.model_selection import IterativeStratification

def data_catch(name):

    """
    Get the data of scikit-multilearn package.
    It used the standart train data as train and test, and the standart test data to validate.

    Parameters:
    --------
        name : string
            The name of a scikit-multilearn package data available
        train_size: float
            The proportion used in train and test model
        random_state: int
            Use the same randon data so that the model can be compared to other ones.
        
    Returns:
    --------
        X_train : numpy
            The data used to train the model
        X_test : numpy
            The data used to test the model
        y_train : numpy
            The label of training data
        y_test : numpy
            The label of testing data
        X_validation : numpy
            The data to validade the model 
        y_validation : numpy
            The labels to validate the model
    """

    X, y, _, _ = load_dataset(name, 'train')
    X_validation, y_validation, _, _ = load_dataset(name, 'test')

    # Data normalization
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X.toarray())
    X_validation = scaler.fit_transform(X_validation.toarray())

    return X, y.toarray(), X_validation, y_validation.toarray()

def model_validation(nome, X_train, y_train, X_validation, y_validation, growing_ideal, smooth_ideal, sf_ideal, alpha_ideal, time_ideal, curve_ideal):
    
    start = time.time()
    init_grid(X_train, sf=sf_ideal, alfa=alpha_ideal)
    start_growing_phase(X_train, growing_ideal)
    start_smoothing_phase(X_train, smooth_ideal)
    end = time.time()
    time_ = end - start

    neuron_labels_list = get_neuron_labels_mutilabel_list(X_train, y_train)
    y_test_predicted = get_labels(X_validation, neuron_labels_list) 

    # Calcular as curvas de precisão-recall para cada classe
    precision, recall, _ = metrics.precision_recall_curve(y_validation.ravel(), np.array(y_test_predicted).ravel())

    # Calcular a AUPRC para cada classe
    auprc = metrics.auc(recall, precision)

    # Calcular a média da AUPRC para todas as classes
    mean_curve = auprc.mean()
    
    print('Time:', time_)
    print(f'Neuron numbers: {len(neuron_labels_list)}')
    print("AUPRC média:", mean_curve)
    performace_measures(y_validation, np.array(y_test_predicted) > 0.5)

    # Criar o gráfico da curva de precisão-recall
    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2, label='Curva de Precisão-Recall (AUC = %0.2f)' % mean_curve)
    plt.xlabel('Recall')
    plt.ylabel('Precisão')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title(str('Curva de Precisão-Recall do ' + nome))
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig('grafico_validacao_'+nome+'.pdf', format='pdf')
    plt.savefig('grafico_validacao_'+nome+'.png')

def find_best_parameters(data_title='emotions', iterations=10, n_growing=10, n_smooting=5, n_sf=10, n_alpha=10, show=False, findFacto=False):
    X, y, X_validation, y_validation = data_catch(data_title)
            
    alpha_list = np.arange(0.005, 0.096, 0.005)   # 19 values
    sf_list = np.arange(0.05, 1, 0.05)             # 20 values

    parameters = find_best_epoche(X, y, n_growing, n_smooting, sf_list, alpha_list, iterations)
    # data: [growing_ideal, smooth_ideal, sf_ideal, alpha_ideal, time_ideal, curve_ideal]
    
    print("Final result:")
    print(parameters)

    model_validation(data_title, X, y, X_validation, y_validation, *parameters)

    return 



def find_best_epoche(X, y, times_growing, times_smooting, sf_list, alpha_list, iterations):

    parameters_list = list()
    best_index = 0
    best_curve = 0

    k_fold = IterativeStratification(n_splits=5, order=1)
    for i, (train, test) in enumerate(k_fold.split(X, y)):
        print(f'\n\n-------------------------\nCross Validation {i+1}:')
        parameters_list.append(find_best_growing_value(X[train], X[test], y[train], y[test], times_growing, times_smooting, sf_list, alpha_list, iterations))
        if (parameters_list[i][-1] > best_curve):
            best_curve = parameters_list[i][-1]
            best_index = i

    print('End of Cross Validation\nAll parameters:', parameters_list, '\n\nChosed:', best_index+1, 'folder')

    return parameters_list[best_index]



def find_best_growing_value(X_train, X_test, y_train, y_test, times_growing, times_smooting, sf_list, alpha_list, iterations, break_code=True):
    print("Entrou no growing")
    parameters_list = list()
    best_index = 0
    best_curve = 0
    
    for j, n_growing in enumerate(range(1, times_growing + 1)):
        print("Fase do Growing:", n_growing)
        parameters_list.append(find_best_smooth_value(X_train, X_test, y_train, y_test, n_growing, times_smooting, sf_list, alpha_list, iterations))
        parameters_list[j].insert(0, n_growing)
        # [n_growing, n_smooth, sf_ideal, alpha_ideal, time_ideal, curve]
        print("Resultado deste Growing:", parameters_list[j])

        if (parameters_list[j][-1] > best_curve):
            best_index = j
            best_curve = parameters_list[j][-1]
        elif ((break_code == True) & (j != 0)):     # So that at least 2 growing parameters exist
            break       # small chance that a algorithm with more growing phases have a better parameter list

    print('Saiu do growing. Resultado:', parameters_list[best_index], '\n')
    return parameters_list[best_index] # [n_growing, n_smooth, sf_ideal, alpha_ideal, size_ideal, proportion_ideal, time_ideal]



def find_best_smooth_value(X_train, X_test, y_train, y_test, n_growing, times_smooting, sf_list, alpha_list, iterations):
    print('Entrou no smooth')
    parameters_list = list()
    best_index = 0
    best_curve = 0

    for j, n_smooth in enumerate(range(1, times_smooting + 1)):
        print("Fase do Smooth:", n_smooth)
        parameters_list.append(find_best_sf(X_train, X_test, y_train, y_test, n_growing, n_smooth, sf_list, alpha_list, iterations))
        parameters_list[j].insert(0, n_smooth)
        # [n_smooth, sf_ideal, alpha_ideal, time_ideal, curve]

        if (parameters_list[j][-1] > best_curve):
            best_index = j
            best_curve = parameters_list[j][-1]
        
    print('Saiu do smooth. Resultado:', parameters_list[best_index], '\n')
    return parameters_list[best_index]
            
def find_best_sf(X_train, X_test, y_train, y_test, n_growing, n_smoothing, sf_list, alpha_list, iterations):

    print("Entrou no sf")
    each_alpha = list()
    comparacao = list()
    comparacao_index = list()

    for (j, sf_atual) in enumerate(sf_list):

        valor_iteracoes = [[X_train, X_test, y_train, y_test, n_growing, n_smoothing, sf_atual, x, iterations] for x in alpha_list]

        pool = multiprocessing.Pool()

        each_alpha.append(pool.map(novo_find_alpha, valor_iteracoes))

        # melhor_valor = max([sub[-1] for sub in each_alpha[j]])
        # comparacao.append([each_alpha[j].index(melhor_valor), melhor_valor])
        # comparacao.append(max(sublista[-1] for lista_exterior in each_alpha for sublista in lista_exterior))      # Pega o caso para o maior alpha
        melhor_caso = 0
        melhor_index = 0
        for (k, item) in enumerate(each_alpha[j]):
            if item[-1] > melhor_caso:
                melhor_caso = item[-1]
                melhor_index = k 
        comparacao.append(melhor_caso)
        comparacao_index.append(melhor_index)

    # print(f'maior valor: {max(comparacao)}')
    # index_melhor_resultado = comparacao.index(max(comparacao))
    index_melhor_externo = comparacao.index(max(comparacao))
    index_melhor_interno = comparacao_index[index_melhor_externo]
    print(each_alpha)
    # print(f'Valor de index externo {index_melhor_externo}')
    # print(f'Valor de index interno {index_melhor_interno}')
    melhor_resultado = each_alpha[index_melhor_externo][index_melhor_interno]
    # melhor_resultado = each_alpha[int(index_melhor_resultado/len(each_alpha))][index_melhor_resultado % len(each_alpha)]    # encontra melhor lista e melhor elemento
    melhor_resultado.insert(0, int(index_melhor_externo + 1) * sf_list[0])       # valor de sf

    # print(f'Melhor resultado: {melhor_resultado}')
    print("Saiu do sf")
    return melhor_resultado

def novo_find_alpha(input):
    # print('Entrou no alpha')
    X_train, X_test, y_train, y_test, n_growing, n_smoothing, n_sf, n_alpha, iterations = input
    time_values = list()
    mean_curve = list()
    for j in range(iterations):

        start = time.time()
        # print('Entrei aqui')
        init_grid(X_train, sf=n_sf, alfa=n_alpha)
        start_growing_phase(X_train, n_growing)
        start_smoothing_phase(X_train, n_smoothing)
        # print('Sai daqui')
        end = time.time()
        time_ = end - start

        neuron_labels_list = get_neuron_labels_mutilabel_list(X_train, y_train, 0.5, include=False)
        y_test_predicted = get_labels(X_test, neuron_labels_list) 

        # Calcular as curvas de precisão-recall para cada classe
        precision, recall, _ = metrics.precision_recall_curve(y_test.ravel(), np.array(y_test_predicted).ravel())

        # Calcular a AUPRC para cada classe
        auprc = metrics.auc(recall, precision)

        # Calcular a média da AUPRC para todas as classes
        mean_curve.append(auprc.mean())
        time_values.append(time_)

    return [n_alpha, s.mean(time_values), s.mean(mean_curve)]