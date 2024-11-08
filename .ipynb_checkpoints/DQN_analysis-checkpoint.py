import matplotlib.pyplot as plt
import pandas as pd

# Lire le fichier CSV
df = pd.read_csv('dqn_training_log.csv')

# Créer des graphiques pour visualiser les données
fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# Graphique 1: Epsilon
axs[0].plot(df['Episode'], df['Epsilon'], color='blue')
axs[0].set_title('Évolution de l\'Epsilon')
axs[0].set_xlabel('Épisode')
axs[0].set_ylabel('Epsilon')

# Graphique 2: Loss (Perte)
axs[1].plot(df['Episode'], df['Loss'], color='red')
axs[1].set_title('Évolution de la Perte')
axs[1].set_xlabel('Épisode')
axs[1].set_ylabel('Loss')

# Graphique 3: Temps de l'épisode
axs[2].plot(df['Episode'], df['Episode Duration'], color='purple')
axs[2].set_title("Évolution du Temps de l'Épisode")
axs[2].set_xlabel('Épisode')
axs[2].set_ylabel('Durée de l\'épisode')

# Ajuster l'affichage
plt.tight_layout()

# Enregistrer la figure comme un fichier PNG
plt.savefig('DQN_log_visualization.png', dpi=300)

plt.show()
