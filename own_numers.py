from matplotlib.pyplot import imread
import NeuralNetworks
import numpy as np

model = NeuralNetworks.FeedForward3(784, 200, 10, 0.01)
model.load_weights("Pretrained/w_ih.csv","Pretrained/w_ho.csv")

img_array = np.absolute((imread("own_numbers/number2.png").flatten()) - 0.99)
output = model.query(img_array).squeeze(axis=1)
label = np.argmax(output)
print(f"Vorhersage: {label}")
print(f"Wahrscheinlichkeit: {round(float(output[label])*100, 2)}%")