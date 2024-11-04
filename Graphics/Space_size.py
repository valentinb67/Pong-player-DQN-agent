import matplotlib.pyplot as plt
import numpy as np

# Intervalle des valeurs de n
x = np.arange(0, 800, 1)

# Définition des fonctions
f = x **3 * 4


# Création du graphique
plt.figure(figsize=(10, 6))
plt.plot(x, f, label=r'$\text{EspaceEtat(dim)} = \text{dim}^3 \times 4$')



# Personnalisation du graphique
plt.xlabel('Dimension')
plt.ylabel("Taille de l'Espace d'Etat")
plt.title("Taille de l'Espace d'Etat en Fonction de la Dimension de l'Environnement")
plt.legend()
plt.grid(True)
plt.savefig("taille_espace_etat.png")
plt.show()