import matplotlib.pyplot as plt
import numpy as np

# Intervalle des valeurs de n
n_values = np.arange(0, 100, 1)  # Par exemple, n varie de 0 à 100

# Définition des fonctions
f1 = 0.9 * 0.96 ** n_values
f2 = 0.9 * 0.97 ** n_values
f3 = 0.9 * 0.975 ** n_values
f4 = 0.9 * 0.980 ** n_values
f5 = 0.9 * 0.985 ** n_values
f6 = 0.9 * 0.990 ** n_values
f7 = 0.9 * 0.995 ** n_values

# Création du graphique
plt.figure(figsize=(10, 6))
plt.plot(n_values, f1, label=r'$epsilonmin6(n) = 0.9 \cdot 0.96^n$')
plt.plot(n_values, f2, label=r'$epsilonmin7(n) = 0.9 \cdot 0.97^n$')
plt.plot(n_values, f3, label=r'$epsilonmin75(n) = 0.9 \cdot 0.975^n$')
plt.plot(n_values, f4, label=r'$epsilonmin80(n) = 0.9 \cdot 0.980^n$')
plt.plot(n_values, f5, label=r'$epsilonmin85(n) = 0.9 \cdot 0.985^n$')
plt.plot(n_values, f6, label=r'$epsilonmin90(n) = 0.9 \cdot 0.990^n$')
plt.plot(n_values, f7, label=r'$epsilonmin95(n) = 0.9 \cdot 0.995^n$')

# Ajout d'une ligne horizontale à y=0.1
plt.axhline(y=0.1, color='gray', linestyle='--', label=r'$y = 0.1$')

# Personnalisation du graphique
plt.xlabel('n_episodes')
plt.ylabel('epsilonmin(n_episodes)')
plt.title('Courbes de décroissance exponentielle représentant différentes fonctions epsilon_min(n) pour différents epsilon_decay')
plt.legend()
plt.grid(True)
plt.show()
