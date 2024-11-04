import matplotlib.pyplot as plt
import pandas as pd

# Lire les fichiers CSV
df_q_learning_discret = pd.read_csv('/Users/valen/Documents/GitHub/Pong-player-DQN-agent/LearningData/1_q_learning_log.csv')
df_q_learning_rdm = pd.read_csv('/Users/valen/Documents/GitHub/Pong-player-DQN-agent/LearningData/1.2_q_learning_rdm_log.csv')
df_q_continuous = pd.read_csv('/Users/valen/Documents/GitHub/Pong-player-DQN-agent/LearningData/1.5_q_learning_continuous_log.csv')
df_dqn_discret = pd.read_csv('/Users/valen/Documents/GitHub/Pong-player-DQN-agent/LearningData/2_pong_dqn_discret_training_log.csv')
df_dqn_continu = pd.read_csv('/Users/valen/Documents/GitHub/Pong-player-DQN-agent/LearningData/2.5_pong_dqn_continuous_training_log.csv')
df_Ddqn_discret = pd.read_csv('/Users/valen/Documents/GitHub/Pong-player-DQN-agent/LearningData/3_pong_double_dqn_discret_training_log.csv')
df_Ddqn_continu = pd.read_csv('/Users/valen/Documents/GitHub/Pong-player-DQN-agent/LearningData/3.5_pong_double_dqn_continuous_training_log.csv')

# Liste des DataFrames et leurs étiquettes pour les légendes
df_dqn = [
    #(df_q_learning_rdm, 'Q-Learning Rdm'),
    #(df_q_learning_discret, 'Q-Learning Discret'),
    #(df_q_continuous, 'Q-Learning Continu'),
    (df_dqn_discret, 'DQN Discret'),
    (df_dqn_continu, 'DQN Continu'),
    (df_Ddqn_discret, 'Double DQN Discret'),
    (df_Ddqn_continu, 'Double DQN Continu')
]

# Définir une palette de couleurs unique
colors = ['blue', 'orange', 'green', 'red']
color_map = {label: color for (_, label), color in zip(df_dqn, colors)}

# Créer des graphiques pour visualiser les données
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Graphique 1: Epsilon
for df, label in df_dqn:
    df_filtered = df[df['Episode'] <= 100]  # Filtrer les épisodes jusqu'à 100
    if 'Epsilon' in df_filtered.columns:
        axs[0].plot(df_filtered['Episode'], df_filtered['Epsilon'], label=label, color=color_map[label])
axs[0].set_title("Évolution de l'Epsilon")
axs[0].set_xlabel("Épisode")
axs[0].set_ylabel("Epsilon")
axs[0].legend()

# Graphique 2: Estimate Value and True Value
for df, label in df_dqn:
    df_filtered = df[df['Episode'] <= 100]  # Filtrer les épisodes jusqu'à 100
    if 'Estimate Value' in df_filtered.columns and 'True Value' in df_filtered.columns:
        axs[1].plot(df_filtered['Episode'], df_filtered['Estimate Value'], label=f"{label} (Estimate)", color=color_map[label])
        axs[1].plot(df_filtered['Episode'], df_filtered['True Value'], label=f"{label} (True)", linestyle='--', color=color_map[label])
axs[1].set_title("Valeurs Estimées et Réelles")
axs[1].set_xlabel("Épisode")
axs[1].set_ylabel("Valeur")
axs[1].legend()

# Graphique 3: Fonction de perte (Loss)
for df, label in df_dqn:
    df_filtered = df[df['Episode'] <= 100]  # Filtrer les épisodes jusqu'à 100
    if 'Loss' in df_filtered.columns:
        axs[2].plot(df_filtered['Episode'], df_filtered['Loss'], label=label, color=color_map[label])
axs[2].set_title('Évolution de la fonction de perte (Loss)')
axs[2].set_xlabel("Épisode")
axs[2].set_ylabel("Loss")
axs[2].set_yscale('log')
axs[2].legend()

# Graphique 4: Durée de l'épisode
#for df, label in df_dqn:
#    df_filtered = df[df['Episode'] <= 100]  # Filtrer les épisodes jusqu'à 100
#    if 'Episode Duration' in df_filtered.columns:
#        axs[3].plot(df_filtered['Episode'], df_filtered['Episode Duration'], label=label, color=color_map[label])
#axs[3].set_title("Evolution de la Durée d'Episode")
#vaxs[3].set_xlabel("Épisode")
#axs[3].set_ylabel("Durée")
#axs[3].set_yscale('log')
#axs[3].legend()

# Graphique 3: Estimate Value and True Value
#for df, label in df_q_discret:
#    df_filtered = df[df['Episode'] <= 100]  # Filtrer les épisodes jusqu'à 100
#    if 'Estimate Value' in df_filtered.columns and 'True Value' in df_filtered.columns:
#        axs[3].plot(df_filtered['Episode'], df_filtered['Estimate Value'], label=f"{label} (Estimate)", color=color_map[label])
#        axs[3].plot(df_filtered['Episode'], df_filtered['True Value'], linestyle='--', color=color_map[label])
#axs[3].set_title("Valeurs Estimées et Réelles")
#axs[3].set_xlabel("Épisode")
#axs[3].set_ylabel("Valeur")
#axs[3].legend()

# Afficher les graphiques
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import pandas as pd

# Lire les fichiers CSV
df_q_learning_discret = pd.read_csv('/Users/valen/Documents/GitHub/Pong-player-DQN-agent/LearningData/1_q_learning_log.csv')
df_q_learning_rdm = pd.read_csv('/Users/valen/Documents/GitHub/Pong-player-DQN-agent/LearningData/1.2_q_learning_rdm_log.csv')
df_q_continuous = pd.read_csv('/Users/valen/Documents/GitHub/Pong-player-DQN-agent/LearningData/1.5_q_learning_continuous_log.csv')
df_dqn_discret = pd.read_csv('/Users/valen/Documents/GitHub/Pong-player-DQN-agent/LearningData/2_pong_dqn_discret_training_log.csv')
df_dqn_continu = pd.read_csv('/Users/valen/Documents/GitHub/Pong-player-DQN-agent/LearningData/2.5_pong_dqn_continuous_training_log.csv')
df_Ddqn_discret = pd.read_csv('/Users/valen/Documents/GitHub/Pong-player-DQN-agent/LearningData/3_pong_double_dqn_discret_training_log.csv')
df_Ddqn_continu = pd.read_csv('/Users/valen/Documents/GitHub/Pong-player-DQN-agent/LearningData/3.5_pong_double_dqn_continuous_training_log.csv')

# Liste des DataFrames et leurs étiquettes pour les légendes
df_dqn_discret = [
    (df_dqn_discret, 'DQN Discret'),
    (df_Ddqn_discret, 'Double DQN Discret')
]

df_dqn_continu = [
    (df_dqn_continu, 'DQN Continu'),
    (df_Ddqn_continu, 'Double DQN Continu')
]

# Définir une palette de couleurs spécifique, bleu pour DQN Discret et Double DQN Discret
color_map = {
    'DQN Discret': 'blue',
    'DQN Continu': 'orange',
    'Double DQN Discret': 'green',
    'Double DQN Continu': 'red'
}

# Graphique 1: Fonction de perte (Loss) pour discret
fig, ax = plt.subplots(figsize=(10, 7))

for df, label in df_dqn_discret:
    df_filtered = df[df['Episode'] <= 100]  # Filtrer les épisodes jusqu'à 100
    if 'Loss' in df_filtered.columns:
        ax.plot(df_filtered['Episode'], df_filtered['Loss'], label=label, color=color_map[label])

ax.set_title('Évolution de la fonction de perte (Loss) en Discret')
ax.set_xlabel("Épisode")
ax.set_ylabel("Loss")
ax.set_yscale('log')
ax.legend()

plt.tight_layout()
plt.show()

# Graphique 2: Estimate Value and True Value pour discret
fig, ax2 = plt.subplots(figsize=(10, 7))

for df, label in df_dqn_discret:
    df_filtered = df[df['Episode'] <= 100]  # Filtrer les épisodes jusqu'à 100
    if 'Estimate Value' in df_filtered.columns and 'True Value' in df_filtered.columns:
        ax2.plot(df_filtered['Episode'], df_filtered['Estimate Value'], label=f"{label} (Estimate)", color=color_map[label])
        ax2.plot(df_filtered['Episode'], df_filtered['True Value'], label=f"{label} (True)", linestyle='--', color=color_map[label])

ax2.set_title('Évolution des Estimate Value and True Value en Discret')
ax2.set_xlabel("Épisode")
ax2.set_ylabel("Value")
ax2.legend()

plt.tight_layout()
plt.show()

# Graphique 3: Fonction de perte (Loss) pour continu
fig, ax = plt.subplots(figsize=(10, 7))

for df, label in df_dqn_continu:
    df_filtered = df[df['Episode'] <= 100]  # Filtrer les épisodes jusqu'à 100
    if 'Loss' in df_filtered.columns:
        ax.plot(df_filtered['Episode'], df_filtered['Loss'], label=label, color=color_map[label])

ax.set_title('Évolution de la fonction de perte (Loss) en Continu')
ax.set_xlabel("Épisode")
ax.set_ylabel("Loss")
ax.legend()

plt.tight_layout()
plt.show()

# Graphique 2: Estimate Value and True Value pour continu
fig, ax2 = plt.subplots(figsize=(10, 7))

for df, label in df_dqn_continu:
    df_filtered = df[df['Episode'] <= 100]  # Filtrer les épisodes jusqu'à 100
    if 'Estimate Value' in df_filtered.columns and 'True Value' in df_filtered.columns:
        ax2.plot(df_filtered['Episode'], df_filtered['Estimate Value'], label=f"{label} (Estimate)", color=color_map[label])
        ax2.plot(df_filtered['Episode'], df_filtered['True Value'], label=f"{label} (True)", linestyle='--', color=color_map[label])

ax2.set_title('Évolution des Estimate Value and True Value en Continu')
ax2.set_xlabel("Épisode")
ax2.set_ylabel("Value")
ax2.legend()

plt.tight_layout()
plt.show()