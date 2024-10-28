#Play vs DDQN continu

import pygame
import sys
import torch
import torch.nn as nn
import numpy as np

# Initialisation de pygame
pygame.init()

# Vérification de la disponibilité de CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dimensions de la fenêtre du jeu
largeur, hauteur = 640, 480
fenetre = pygame.display.set_mode((largeur, hauteur))
pygame.display.set_caption("Pong - Jouer contre l'IA")

# Couleurs
blanc = (255, 255, 255)

# Initialisation des raquettes et de la balle
raquette1 = pygame.Rect(50, hauteur // 2 - 70, 10, 140)
raquette2 = pygame.Rect(largeur - 60, hauteur // 2 - 70, 10, 140)
balle = pygame.Rect(largeur // 2 - 15, hauteur // 2 - 15, 30, 30)

# Vitesses initiales de la balle
vitesse_balle_x = -12
vitesse_balle_y = 6 * np.random.choice([-1, 1])

# Initialisation des scores
score_joueur1 = 0
score_joueur2 = 0

# Police d'affichage des scores
font = pygame.font.Font(None, 36)

# Modèle de réseau de neurones pour DQN
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(5, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)  # 3 actions : UP, DOWN, STAY

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Charger le modèle entraîné
policy_net = DQN().to(device)
policy_net.load_state_dict(torch.load('5_Ddqn_continuous.pth', map_location=device))
policy_net.eval()

def obtenir_etat_continu():
    etat_x = balle.x / largeur
    etat_y = balle.y / hauteur
    etat_vx = vitesse_balle_x / 12.0
    etat_vy = vitesse_balle_y / 12.0
    raquette_pos = raquette2.y / hauteur
    return np.array([etat_x, etat_y, etat_vx, etat_vy, raquette_pos], dtype=np.float32)

def choisir_action(state):
    with torch.no_grad():
        state_tensor = torch.tensor(state).to(device)
        return policy_net(state_tensor).argmax().item()

def reinitialiser_jeu():
    global balle, vitesse_balle_x, vitesse_balle_y
    balle = pygame.Rect(largeur // 2 - 15, hauteur // 2 - 15, 30, 30)
    vitesse_balle_x = -12
    vitesse_balle_y = 6 * np.random.choice([-1, 1])

# Boucle principale
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    state = obtenir_etat_continu()

    # Contrôle de la raquette du joueur 1
    keys = pygame.key.get_pressed()
    if keys[pygame.K_z] and raquette1.top > 0:
        raquette1.y -= 10
    if keys[pygame.K_s] and raquette1.bottom < hauteur:
        raquette1.y += 10

    # Action de l'IA pour la raquette du joueur 2
    action_idx = choisir_action(state)
    if action_idx == 0 and raquette2.top > 0:
        raquette2.y -= 10
    elif action_idx == 1 and raquette2.bottom < hauteur:
        raquette2.y += 10

    # Mise à jour de la position de la balle
    balle.x += vitesse_balle_x
    balle.y += vitesse_balle_y

    # Collision avec le haut ou le bas de l'écran
    if balle.top <= 0 or balle.bottom >= hauteur:
        vitesse_balle_y = -vitesse_balle_y

    # Collision avec les raquettes
    if balle.colliderect(raquette1):
        vitesse_balle_x = -vitesse_balle_x
        balle.left = raquette1.right
    if balle.colliderect(raquette2):
        vitesse_balle_x = -vitesse_balle_x
        balle.right = raquette2.left

    # Mise à jour des scores et réinitialisation si la balle sort de l'écran
    if balle.left <= 0:
        score_joueur2 += 1
        reinitialiser_jeu()
    if balle.right >= largeur:
        score_joueur1 += 1
        reinitialiser_jeu()

    # Dessin de l'écran de jeu
    fenetre.fill((0, 0, 0))
    pygame.draw.rect(fenetre, blanc, raquette1)
    pygame.draw.rect(fenetre, blanc, raquette2)
    pygame.draw.ellipse(fenetre, blanc, balle)

    # Affichage des scores
    score_text = font.render(f"Joueur 1: {score_joueur1}  Joueur 2: {score_joueur2}", True, blanc)
    fenetre.blit(score_text, (largeur // 2 - 100, 10))

    pygame.display.flip()
    pygame.time.Clock().tick(60)
