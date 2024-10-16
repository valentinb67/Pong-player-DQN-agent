import pygame
import sys
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# Initialisation de pygame
pygame.init()

# Dimensions de la fenêtre du jeu
largeur, hauteur = 640, 480
fenetre = pygame.display.set_mode((largeur, hauteur))
pygame.display.set_caption("Pong")

# Couleurs
blanc = (255, 255, 255)

# Initialisation des raquettes et de la balle
raquette1 = pygame.Rect(50, hauteur // 2 - 70, 10, 140)  # Raquette du joueur
raquette2 = pygame.Rect(largeur - 60, hauteur // 2 - 70, 10, 140)  # Raquette de l'ordinateur
balle = pygame.Rect(largeur // 2 - 15, hauteur // 2 - 15, 30, 30)  # Balle au centre

# Vitesses initiales de la balle
vitesse_balle_x = -12
vitesse_balle_y = 12 * random.choice([-1, 1])

# Hyperparamètres DQN
alpha = 0.001  # Taux d'apprentissage
gamma = 0.99  # Facteur de réduction
epsilon = 0.9  # Exploration initiale
epsilon_decay = 0.995  # Décroissance d'epsilon
epsilon_min = 0.1  # Valeur minimale d'epsilon
batch_size = 64
memory_size = 10000  # Taille du buffer de replay
target_update = 10  # Fréquence de mise à jour du réseau cible
memory = deque(maxlen=memory_size)

# Paramètres du jeu
actions = ["UP", "DOWN", "STAY"]
nb_actions = len(actions)

# Modèle de réseau de neurones pour DQN
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(5, 128)  # Entrée : 5 (état)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, nb_actions)  # Sortie : nb_actions (Q-values)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Retourne Q-values pour chaque action

# Instanciation des réseaux principal et cible
policy_net = DQN()
target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())  # Réseau cible est initialisé avec les mêmes poids
target_net.eval()  # Le réseau cible n'est pas entraîné

# Optimiseur
optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
loss_fn = nn.MSELoss()

# Fonction pour discrétiser l'état
def discretiser(val, intervalle, nb_divisions):
    return min(nb_divisions - 1, max(0, int(val / intervalle * nb_divisions)))

def obtenir_etat_discret():
    etat_x = discretiser(balle.x, largeur, 20)
    etat_y = discretiser(balle.y, hauteur, 20)
    etat_vx = 0 if vitesse_balle_x < 0 else 1
    etat_vy = 0 if vitesse_balle_y < 0 else 1
    raquette_pos = discretiser(raquette2.y, hauteur, 20)
    return np.array([etat_x, etat_y, etat_vx, etat_vy, raquette_pos], dtype=np.float32)

# Fonction pour choisir une action (epsilon-greedy)
def choisir_action(state):
    global epsilon
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(nb_actions))  # Exploration
    else:
        with torch.no_grad():
            return policy_net(torch.tensor(state)).argmax().item()  # Exploitation

# Fonction pour stocker la transition dans le buffer de replay
def stocker_transition(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

# Fonction de mise à jour du DQN
def entrainer_dqn():
    if len(memory) < batch_size:
        return

    # Échantillonner un batch de transitions depuis la mémoire
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    # Conversion en tenseurs PyTorch
    states = torch.tensor(states)
    actions = torch.tensor(actions).unsqueeze(1)
    rewards = torch.tensor(rewards)
    next_states = torch.tensor(next_states)
    dones = torch.tensor(dones, dtype=torch.float32)

    # Calcul des valeurs Q actuelles pour les actions prises
    q_values = policy_net(states).gather(1, actions).squeeze()

    # Calcul des valeurs Q cibles
    next_q_values = target_net(next_states).max(1)[0]
    expected_q_values = rewards + (gamma * next_q_values * (1 - dones))

    # Calcul de la perte et optimisation
    loss = loss_fn(q_values, expected_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Fonction pour réinitialiser l'environnement après chaque épisode
def reinitialiser_jeu():
    global balle, vitesse_balle_x, vitesse_balle_y
    balle = pygame.Rect(largeur // 2 - 15, hauteur // 2 - 15, 30, 30)
    vitesse_balle_x = -12
    vitesse_balle_y = 12 * random.choice([-1, 1])

# Boucle principale du jeu
frames = 0
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Réduire epsilon (exploration)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Récupérer l'état actuel
    state = obtenir_etat_discret()

    # Mouvement de la raquette du joueur 1
    raquette1.y = balle.y
    
    # Choisir une action
    action_idx = choisir_action(state)
    action = actions[action_idx]

    # Exécuter l'action
    if action == "UP" and raquette2.top > 0:
        raquette2.y -= 10
    elif action == "DOWN" and raquette2.bottom < hauteur:
        raquette2.y += 10

    # Déplacement de la balle
    balle.x += vitesse_balle_x
    balle.y += vitesse_balle_y

    # Rebond de la balle sur les bords
    if balle.top <= 0 or balle.bottom >= hauteur:
        vitesse_balle_y = -vitesse_balle_y

    # Récompense et nouvelle transition
    reward = 0
    if balle.colliderect(raquette1) or balle.colliderect(raquette2):
        vitesse_balle_x = -vitesse_balle_x
        reward = 1

    done = False
    if balle.left <= 0:  # Si la balle sort du côté du joueur
        reward = -10
        done = True
        reinitialiser_jeu()

    if balle.right >= largeur:  # Si la balle sort du côté de l'ordinateur
        reward = 10
        done = True
        reinitialiser_jeu()

    # Récupérer le nouvel état
    next_state = obtenir_etat_discret()

    # Stocker la transition
    stocker_transition(state, action_idx, reward, next_state, done)

    # Entraîner le DQN
    entrainer_dqn()

    # Mettre à jour le réseau cible tous les `target_update` épisodes
    frames += 1
    if frames % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Effacer l'écran
    fenetre.fill((0, 0, 0))

    # Dessiner les raquettes et la balle
    pygame.draw.rect(fenetre, blanc, raquette1)
    pygame.draw.rect(fenetre, blanc, raquette2)
    pygame.draw.ellipse(fenetre, blanc, balle)

    # Rafraîchir l'écran
    pygame.display.flip()

    # Limite de rafraîchissement
    pygame.time.Clock().tick(60)
