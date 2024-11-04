import pygame
import sys
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import csv
import time  # Pour mesurer la durée de l'épisode

# Initialisation de pygame
pygame.init()

# Vérification de la disponibilité de CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dimensions de la fenêtre du jeu
largeur, hauteur = 640, 480
fenetre = pygame.display.set_mode((largeur, hauteur))
pygame.display.set_caption("Pong")

# Couleurs
blanc = (255, 255, 255)

# Initialisation des raquettes et de la balle
raquette1 = pygame.Rect(50, hauteur // 2 - 70, 10, 140)
raquette2 = pygame.Rect(largeur - 60, hauteur // 2 - 70, 10, 140)
balle = pygame.Rect(largeur // 2 - 15, hauteur // 2 - 15, 30, 30)

# Vitesses initiales de la balle
vitesse_balle_x = -12
vitesse_balle_y = 6 * random.choice([-1, 1])

# Initialisation des scores
score_joueur1 = 0
score_joueur2 = 0

# Police d'affichage des scores et autres informations
font = pygame.font.Font(None, 36)

# Initialisation pour l'enregistrement vidéo
fps = 60
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('Records/3.5_Ddqn_record.avi', fourcc, fps, (largeur, hauteur))

# Hyperparamètres Double DQN
alpha = 0.001
gamma = 0.99
epsilon = 0.9
epsilon_decay = 0.975
epsilon_min = 0.1
batch_size = 128
memory_size = 10000
target_update = 10

# Nbr d'épisodes
max_episodes = 100
episode_count = 0

memory = deque(maxlen=memory_size)

# Paramètres du jeu
nb_actions = 3  # UP, DOWN, STAY

# Variables pour l'enregistrement des données d'entraînement
csv_file = open('LearningData/3.5_pong_double_dqn_continuous_training_log.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Episode', 'Epsilon', 'Reward', 'Reward cumulee', 'Episode Duration', 'Loss', 'True Value', 'Estimate Value', 'TD Error'])

# Modèle de réseau de neurones pour DQN
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(5, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, nb_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Instanciation des réseaux principal et cible
policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Optimiseur
optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
loss_fn = nn.MSELoss()

def obtenir_etat_continu():
    etat_x = balle.x / largeur
    etat_y = balle.y / hauteur
    etat_vx = vitesse_balle_x / 12.0
    etat_vy = vitesse_balle_y / 12.0
    raquette_pos = raquette2.y / hauteur
    return np.array([etat_x, etat_y, etat_vx, etat_vy, raquette_pos], dtype=np.float32)

def choisir_action(state):
    global epsilon
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(nb_actions))
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state).to(device)
            return policy_net(state_tensor).argmax().item()

def stocker_transition(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def entrainer_ddqn():
    if len(memory) < batch_size:
        return 0, [], [], []  # Pas d'entraînement si la mémoire est insuffisante

    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states).to(device)
    actions = torch.tensor(actions).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards).to(device)
    next_states = torch.tensor(next_states).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    # Double DQN
    q_values = policy_net(states).gather(1, actions).squeeze()
    next_actions = policy_net(next_states).argmax(1).unsqueeze(1) #L'ajout de cette ligne nous prouve le DOUBLE DQN
    next_q_values = target_net(next_states).gather(1, next_actions).squeeze()
    expected_q_values = rewards + (gamma * next_q_values * (1 - dones))

    loss = loss_fn(q_values, expected_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Calculer les valeurs True Value, Estimate Value et TD Error
    true_values = expected_q_values.detach().cpu().numpy()
    estimate_values = q_values.detach().cpu().numpy()
    td_errors = true_values - estimate_values

    return loss.item(), true_values, estimate_values, td_errors  # Retourner la perte et les nouvelles valeurs pour le suivi

def reinitialiser_jeu():
    global balle, vitesse_balle_x, vitesse_balle_y
    balle = pygame.Rect(largeur // 2 - 15, hauteur // 2 - 15, 30, 30)
    vitesse_balle_x = -12
    vitesse_balle_y = 6 * random.choice([-1, 1])

# Boucle principale
frames = 0
while episode_count < max_episodes:
    episode_reward = 0
    episode_loss = 0  # Suivi de la perte totale pour l'épisode
    start_time = time.time()  # Début de l'épisode

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # Enregistrer le modèle avant de quitter
            torch.save(policy_net.state_dict(), 'ModelsPTH/3.5_Ddqn_continuous.pth')
            video_writer.release()
            pygame.quit()
            sys.exit()

    state = obtenir_etat_continu()

    raquette1.y = balle.y
    
    action_idx = choisir_action(state)
    if action_idx == 0 and raquette2.top > 0:
        raquette2.y -= 10
    elif action_idx == 1 and raquette2.bottom < hauteur:
        raquette2.y += 10

    balle.x += vitesse_balle_x
    balle.y += vitesse_balle_y

    if balle.top <= 0 or balle.bottom >= hauteur:
        vitesse_balle_y = -vitesse_balle_y

    reward = 0
    
    if balle.colliderect(raquette1):
        vitesse_balle_x = -vitesse_balle_x
        balle.left = raquette1.right
        
    if balle.colliderect(raquette2):
        vitesse_balle_x = -vitesse_balle_x
        balle.right = raquette2.left
        reward = 1

    done = False
    if balle.left <= 0:
        score_joueur2 += 1  # Mise à jour du score du joueur 2
        reward = -10
        done = True
        reinitialiser_jeu()
        episode_count += 1

    if balle.right >= largeur:
        score_joueur1 += 1  # Mise à jour du score du joueur 1
        reward = -10
        done = True
        reinitialiser_jeu()
        episode_count += 1

    episode_reward += reward  # Ajout de la récompense au total de l'épisode

    next_state = obtenir_etat_continu()
    stocker_transition(state, action_idx, reward, next_state, done)
    loss, true_values, estimate_values, td_errors = entrainer_ddqn()
    episode_loss += loss  # Ajout de la perte pour chaque batch

    frames += 1
    if frames % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    fenetre.fill((0, 0, 0))

    pygame.draw.rect(fenetre, blanc, raquette1)
    pygame.draw.rect(fenetre, blanc, raquette2)
    pygame.draw.ellipse(fenetre, blanc, balle)

    # Affichage des scores
    score_text = font.render(f"Joueur 1: {score_joueur1}  Joueur 2: {score_joueur2}", True, blanc)
    fenetre.blit(score_text, (largeur // 2 - 100, 10))

    # Affichage de l'épisode et de la valeur epsilon en cours
    episode_text = font.render(f"Episode: {episode_count}  Epsilon: {epsilon:.3f}", True, blanc)
    fenetre.blit(episode_text, (10, hauteur - 40))

    pygame.display.flip()
    pygame.time.Clock().tick(60)
    
    # Capture de l'écran pour l'enregistrement vidéo
    frame = pygame.surfarray.array3d(pygame.display.get_surface())
    frame = cv2.transpose(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video_writer.write(frame)

    # Limite de rafraîchissement
    pygame.time.Clock().tick(fps)

    # Si l'épisode est terminé, on enregistre les informations dans le CSV
    if done:
        episode_duration = time.time() - start_time
        for true_value, estimate_value, td_error in zip(true_values, estimate_values, td_errors):
            csv_writer.writerow([episode_count, epsilon, reward, episode_reward, episode_duration, episode_loss, true_value, estimate_value, td_error])

        # Decay sur epsilon à la fin de l'épisode
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
# Enregistrer le modèle après l'entraînement
torch.save(policy_net.state_dict(), 'ModelsPTH/3.5_Ddqn_continuous.pth')

# Fermer le fichier CSV après l'entraînement
csv_file.close()
video_writer.release()
pygame.quit()
sys.exit()
