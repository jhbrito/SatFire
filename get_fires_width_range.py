import os
import matplotlib.pyplot as plt
import numpy as np

tamanhos = []

if os.path.isfile('./maximum_shapes_width.txt'):
    with open('maximum_shapes_width.txt', 'r+') as f:
        tamanhos = [float(x) for x in f.read().split()]

N = len(tamanhos)
print(N)
numero_de_incendios=dict()

m=0

for limite in range(300, 1000, 1):
    n=0
    limite_minimo=limite/3
    for i in range(len(tamanhos)):
        tam=tamanhos[i] * 1000 #convert from meters to km
        if tam>limite_minimo and tam<limite:
            n=n+1
    numero_de_incendios[limite]=n

for limite in range(300, 1000, 1):
    print(str(limite) + " " + str(numero_de_incendios[limite]))

print(max(numero_de_incendios.values()))
l=list(numero_de_incendios.values())
plt.plot(l)
plt.title('Number of fires according to their maximum width')
plt.xlabel('Fires Width')
plt.ylabel('Number of Fires')

plt.show()