import matplotlib.pyplot as mpl
import numpy as np
import scipy.ndimage
import NeuralNetworks
import scipy
import pandas

model = NeuralNetworks.FeedForward3(784, 200, 10, 0.01)
model.init_weights()

def print_model(model: NeuralNetworks.FeedForward3):
    print('====================================================================================================')
    print('- Schichten: 3')
    print(f'- Lernrate: {model.l_rate}')
    print(f'- Neuronen insgesamt: {model.i_nodes + model.h_nodes + model.o_nodes}')
    print(f'- Gewichte insgesamt: {model.w_ih.size + model.w_ho.size}')
    print(f'- Eingabe-Neuronen: {model.i_nodes}')
    print(f'- Hidden-Neuronen: {model.h_nodes}')
    print(f'- Output-Neuronen: {model.o_nodes}')

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r({iteration}/{total}) Bilder verarbeitet {prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    
    # Print New Line on Complete
    if iteration == total: 
        print()

def train_network(dataset_path, epochen):
    print('====================================================================================================')
    print('[Training]')
    print(f'- Epochen: {epochen}')
    # Trainingsdatenset laden
    df = pandas.read_csv(dataset_path, header=None)

    length = len(df.index) * 3 * epochen
    printProgressBar(0, length, suffix = 'Trainingsfortschritt', length = 50)
    
    for e in range(epochen):
        for i in range(len(df.index)):
            # Einzelne Werte der CSV in einen Array einlesen
            all_values = df.iloc[i]
            # Eingabe formatieren
            inputs = (np.asarray(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01
            # Leicht gedrehte Bildvarianten erstellen
            inputs_plus10_img = scipy.ndimage.rotate(inputs.reshape(28,28), 10, cval=0.01, reshape=False).flatten()
            inputs_minus10_img = scipy.ndimage.rotate(inputs.reshape(28,28), -10, cval=0.01, reshape=False).flatten()
            targets = np.zeros(10) + 0.01
            targets[int(all_values[0])] = 0.99
            model.train(inputs, targets)
            printProgressBar(iteration=((i + 1) + e * len(df.index)), total=length, suffix = 'Trainingsfortschritt', length = 50)
            model.train(inputs_plus10_img, targets)
            printProgressBar(iteration=((i + 1) + e * len(df.index) + 1), total=length, suffix = 'Trainingsfortschritt', length = 50)
            model.train(inputs_minus10_img, targets)
            printProgressBar(iteration=((i + 1) + e * len(df.index) + 2), total=length, suffix = 'Trainingsfortschritt', length = 50)
    print('----------------------------------------------------------------------------------------------------')

def test_network(dataset_path):
    print('[Performence-Test]')
    # Testdatenset laden
    dataset_file = open(dataset_path, "r")
    dataset_list = dataset_file.readlines()
    dataset_file.close()

    scorecard = []

    for record in dataset_list:
        all_values = record.split(",")
        correct_label = int(all_values[0])
        inputs = (np.asarray(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01
        output = model.query(inputs)
        label = np.argmax(output)

        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)

    scorecard = np.asarray(scorecard)
    performence = round(scorecard.sum() / scorecard.size, 4)*100
    print(f"- Korrekt erkannte Bilder: {scorecard.sum()}/{scorecard.size}")
    print(f"- Performence: {performence}%")
    print('====================================================================================================')

def save_weights():
    np.savetxt("w_ih.csv", model.w_ih, delimiter=",")
    np.savetxt("w_ho.csv", model.w_ho, delimiter=",")

print_model(model)
train_network("MNIST-Dataset/mnist_train.csv", 10)
test_network("MNIST-Dataset/mnist_test.csv")
save_weights()

