import matplotlib.pyplot as plt
import pandas as pd

# Lire le fichier CSV
df = pd.read_csv('pong_dqn_discret_training_log.csv')

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

#pong_double_dqn_discret_training_log.csv

# Lire le fichier CSV
df2 = pd.read_csv('pong_double_dqn_discret_training_log.csv')

# Créer des graphiques pour visualiser les données
fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# Graphique 1: Epsilon
axs[0].plot(df2['Episode'], df2['Epsilon'], color='blue')
axs[0].set_title('Évolution de l\'Epsilon')
axs[0].set_xlabel('Épisode')
axs[0].set_ylabel('Epsilon')

# Graphique 2: Loss (Perte)
axs[1].plot(df2['Episode'], df2['Loss'], color='red')
axs[1].set_title('Évolution de la Perte')
axs[1].set_xlabel('Épisode')
axs[1].set_ylabel('Loss')

# Graphique 3: Temps de l'épisode
axs[2].plot(df2['Episode'], df2['Episode Duration'], color='purple')
axs[2].set_title("Évolution du Temps de l'Épisode")
axs[2].set_xlabel('Épisode')
axs[2].set_ylabel('Durée de l\'épisode')

# Ajuster l'affichage
plt.tight_layout()

# Enregistrer la figure comme un fichier PNG
plt.savefig('DQN_log_visualization.png', dpi=300)

plt.show()


# Lire le fichier CSV
df3 = pd.read_csv('pong_ddqn_test.csv')

# Créer des graphiques pour visualiser les données
fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# Graphique 1: Epsilon
axs[0].plot(df3['Episode'], df3['Epsilon'], color='blue')
axs[0].set_title('Évolution de l\'Epsilon')
axs[0].set_xlabel('Épisode')
axs[0].set_ylabel('Epsilon')

# Graphique 2: Loss (Perte)
axs[1].plot(df3['Episode'], df3['Loss'], color='red')
axs[1].set_title('Évolution de la Perte')
axs[1].set_xlabel('Épisode')
axs[1].set_ylabel('Loss')

# Graphique 3: Temps de l'épisode
axs[2].plot(df3['Episode'], df3['Episode Duration'], color='purple')
axs[2].set_title("Évolution du Temps de l'Épisode")
axs[2].set_xlabel('Épisode')
axs[2].set_ylabel('Durée de l\'épisode')

# Ajuster l'affichage
plt.tight_layout()

# Enregistrer la figure comme un fichier PNG
plt.savefig('DQN_log_visualization.png', dpi=300)

plt.show()

import matplotlib.pyplot as plt
import pandas as pd

# Charger les données des trois fichiers CSV
df1 = pd.read_csv('pong_dqn_discret_training_log.csv')
df2 = pd.read_csv('pong_double_dqn_discret_training_log.csv')
df3 = pd.read_csv('pong_ddqn_test.csv')

# Créer une figure avec 3 lignes et 3 colonnes
fig, axs = plt.subplots(3, 3, figsize=(15, 15))

# Première colonne : Données du premier fichier
axs[0, 0].plot(df1['Episode'], df1['Epsilon'], color='blue')
axs[0, 0].set_title("Évolution de l'Epsilon (DQN)")
axs[0, 0].set_xlabel('Épisode')
axs[0, 0].set_ylabel('Epsilon')

axs[1, 0].plot(df1['Episode'], df1['Loss'], color='red')
axs[1, 0].set_title("Évolution de la Perte (DQN)")
axs[1, 0].set_xlabel('Épisode')
axs[1, 0].set_ylabel('Loss')

axs[2, 0].plot(df1['Episode'], df1['Episode Duration'], color='purple')
axs[2, 0].set_title("Évolution du Temps de l'Épisode (DQN)")
axs[2, 0].set_xlabel('Épisode')
axs[2, 0].set_ylabel("Durée de l'épisode")

# Deuxième colonne : Données du deuxième fichier
axs[0, 1].plot(df2['Episode'], df2['Epsilon'], color='blue')
axs[0, 1].set_title("Évolution de l'Epsilon (Double DQN)")
axs[0, 1].set_xlabel('Épisode')
axs[0, 1].set_ylabel('Epsilon')

axs[1, 1].plot(df2['Episode'], df2['Loss'], color='red')
axs[1, 1].set_title("Évolution de la Perte (Double DQN)")
axs[1, 1].set_xlabel('Épisode')
axs[1, 1].set_ylabel('Loss')

axs[2, 1].plot(df2['Episode'], df2['Episode Duration'], color='purple')
axs[2, 1].set_title("Évolution du Temps de l'Épisode (Double DQN)")
axs[2, 1].set_xlabel('Épisode')
axs[2, 1].set_ylabel("Durée de l'épisode")

# Troisième colonne : Données du troisième fichier
axs[0, 2].plot(df3['Episode'], df3['Epsilon'], color='blue')
axs[0, 2].set_title("Évolution de l'Epsilon (DDQN Test)")
axs[0, 2].set_xlabel('Épisode')
axs[0, 2].set_ylabel('Epsilon')

axs[1, 2].plot(df3['Episode'], df3['Loss'], color='red')
axs[1, 2].set_title("Évolution de la Perte (DDQN Test)")
axs[1, 2].set_xlabel('Épisode')
axs[1, 2].set_ylabel('Loss')

axs[2, 2].plot(df3['Episode'], df3['Episode Duration'], color='purple')
axs[2, 2].set_title("Évolution du Temps de l'Épisode (DDQN Test)")
axs[2, 2].set_xlabel('Épisode')
axs[2, 2].set_ylabel("Durée de l'épisode")

# Ajuster l'affichage global
plt.tight_layout()

# Enregistrer la figure comme un fichier PNG
plt.savefig('DQN_combined_log_visualization.png', dpi=300)

# Afficher la figure
plt.show()

import matplotlib.pyplot as plt
import pandas as pd

# Charger les données des trois fichiers CSV
df1 = pd.read_csv('pong_dqn_discret_training_log.csv')
df2 = pd.read_csv('pong_double_dqn_discret_training_log.csv')
df3 = pd.read_csv('pong_ddqn_test.csv')

# Créer une figure avec 6 lignes et 3 colonnes pour accueillir les nouveaux graphiques
fig, axs = plt.subplots(6, 3, figsize=(15, 25))

# Colonne 1 : Données du premier fichier
axs[0, 0].plot(df1['Episode'], df1['Epsilon'], color='blue')
axs[0, 0].set_title("Évolution de l'Epsilon (DQN)")
axs[0, 0].set_xlabel('Épisode')
axs[0, 0].set_ylabel('Epsilon')

axs[1, 0].plot(df1['Episode'], df1['Loss'], color='red')
axs[1, 0].set_title("Évolution de la Perte (DQN)")
axs[1, 0].set_xlabel('Épisode')
axs[1, 0].set_ylabel('Loss')

axs[2, 0].plot(df1['Episode'], df1['Episode Duration'], color='purple')
axs[2, 0].set_title("Évolution du Temps de l'Épisode (DQN)")
axs[2, 0].set_xlabel('Épisode')
axs[2, 0].set_ylabel("Durée de l'épisode")

axs[3, 0].plot(df1['Episode'], df1['True Value'], color='green')
axs[3, 0].set_title("True Value (DQN)")
axs[3, 0].set_xlabel('Épisode')
axs[3, 0].set_ylabel("True Value")

axs[4, 0].plot(df1['Episode'], df1['Value Estimate'], color='orange')
axs[4, 0].set_title("Value Estimate (DQN)")
axs[4, 0].set_xlabel('Épisode')
axs[4, 0].set_ylabel("Value Estimate")

axs[5, 0].plot(df1['Episode'], df1['TD Error'], color='brown')
axs[5, 0].set_title("TD Error (DQN)")
axs[5, 0].set_xlabel('Épisode')
axs[5, 0].set_ylabel("TD Error")

# Colonne 2 : Données du deuxième fichier
axs[0, 1].plot(df2['Episode'], df2['Epsilon'], color='blue')
axs[0, 1].set_title("Évolution de l'Epsilon (Double DQN)")
axs[0, 1].set_xlabel('Épisode')
axs[0, 1].set_ylabel('Epsilon')

axs[1, 1].plot(df2['Episode'], df2['Loss'], color='red')
axs[1, 1].set_title("Évolution de la Perte (Double DQN)")
axs[1, 1].set_xlabel('Épisode')
axs[1, 1].set_ylabel('Loss')

axs[2, 1].plot(df2['Episode'], df2['Episode Duration'], color='purple')
axs[2, 1].set_title("Évolution du Temps de l'Épisode (Double DQN)")
axs[2, 1].set_xlabel('Épisode')
axs[2, 1].set_ylabel("Durée de l'épisode")

axs[3, 1].plot(df2['Episode'], df2['True Value'], color='green')
axs[3, 1].set_title("True Value (Double DQN)")
axs[3, 1].set_xlabel('Épisode')
axs[3, 1].set_ylabel("True Value")

axs[4, 1].plot(df2['Episode'], df2['Value Estimate'], color='orange')
axs[4, 1].set_title("Value Estimate (Double DQN)")
axs[4, 1].set_xlabel('Épisode')
axs[4, 1].set_ylabel("Value Estimate")

axs[5, 1].plot(df2['Episode'], df2['TD Error'], color='brown')
axs[5, 1].set_title("TD Error (Double DQN)")
axs[5, 1].set_xlabel('Épisode')
axs[5, 1].set_ylabel("TD Error")

# Colonne 3 : Données du troisième fichier
axs[0, 2].plot(df3['Episode'], df3['Epsilon'], color='blue')
axs[0, 2].set_title("Évolution de l'Epsilon (DDQN Test)")
axs[0, 2].set_xlabel('Épisode')
axs[0, 2].set_ylabel('Epsilon')

axs[1, 2].plot(df3['Episode'], df3['Loss'], color='red')
axs[1, 2].set_title("Évolution de la Perte (DDQN Test)")
axs[1, 2].set_xlabel('Épisode')
axs[1, 2].set_ylabel('Loss')

axs[2, 2].plot(df3['Episode'], df3['Episode Duration'], color='purple')
axs[2, 2].set_title("Évolution du Temps de l'Épisode (DDQN Test)")
axs[2, 2].set_xlabel('Épisode')
axs[2, 2].set_ylabel("Durée de l'épisode")

axs[3, 2].plot(df3['Episode'], df3['True Value'], color='green')
axs[3, 2].set_title("True Value (DDQN Test)")
axs[3, 2].set_xlabel('Épisode')
axs[3, 2].set_ylabel("True Value")

axs[4, 2].plot(df3['Episode'], df3['Value Estimate'], color='orange')
axs[4, 2].set_title("Value Estimate (DDQN Test)")
axs[4, 2].set_xlabel('Épisode')
axs[4, 2].set_ylabel("Value Estimate")

axs[5, 2].plot(df3['Episode'], df3['TD Error'], color='brown')
axs[5, 2].set_title("TD Error (DDQN Test)")
axs[5, 2].set_xlabel('Épisode')
axs[5, 2].set_ylabel("TD Error")

# Ajuster l'affichage global
plt.tight_layout()

# Enregistrer la figure comme un fichier PNG
plt.savefig('DQN_combined_log_visualization_with_additional_metrics.png', dpi=300)

# Afficher la figure
plt.show()


