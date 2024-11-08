#Play vs DDQN discret

import pygame
import sys
import torch
import torch.nn as nn
import numpy as np

# Initialisation de pygame
pygame.init()

# Vérification de la disponibilité de CUDA pour l'usage de GPU si disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dimensions de la fenêtre du jeu
largeur, hauteur = 640, 480
fenetre = pygame.display.set_mode((largeur, hauteur))
pygame.display.set_caption("Pong - Jouer contre l'IA")

# Couleurs
blanc = (255, 255, 255)

# Initialisation des raquettes et de la balle
# raquette1 est la raquette du joueur humain, raquette2 celle de l'IA
raquette1 = pygame.Rect(50, hauteur // 2 - 70, 10, 140)
raquette2 = pygame.Rect(largeur - 60, hauteur // 2 - 70, 10, 140)
balle = pygame.Rect(largeur // 2 - 15, hauteur // 2 - 15, 30, 30)

# Vitesses initiales de la balle
vitesse_balle_x = -12
vitesse_balle_y = 6 * np.random.choice([-1, 1]) # Direction aléatoire de la vitesse verticale

# Initialisation des scores
score_joueur1 = 0
score_joueur2 = 0

# Police d'affichage des scores
font = pygame.font.Font(None, 36)

# Modèle de réseau de neurones pour DQN
class DQN(nn.Module):
    def __init__(self):
        """
        Initialise un réseau de neurones DQN avec trois couches entièrement connectées.
        
        Le réseau prend en entrée 5 paramètres :
        - Position x de la balle
        - Position y de la balle
        - Vitesse x de la balle
        - Vitesse y de la balle
        - Position de la raquette IA
        
        Return : 3 actions possibles (Monter, Descendre, Rester).
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(5, 128) # Première couche cachée de 128 neurones
        self.fc2 = nn.Linear(128, 128) # Deuxième couche cachée de 128 neurones
        self.fc3 = nn.Linear(128, 3)  # Sortie : 3 actions possibles (Monter, Descendre, Rester)

    def forward(self, x):
        """
        Définition de la passe avant (forward pass) du réseau de neurones.
        Args:
            x (Tensor): Les valeurs d'entrée représentant l'état actuel.
        Return:
            Tensor: Les valeurs de sortie correspondant aux scores des actions possibles.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Charger le modèle pré-entraîné
policy_net = DQN().to(device)
policy_net.load_state_dict(torch.load('ModelsPTH/2_dqn_discret.pth', map_location=device))
policy_net.eval()

def discretiser(val, intervalle, nb_divisions):
    """
    Discrétise une valeur donnée en fonction d'un intervalle et d'un nombre de divisions.
    Args:
        val (float): La valeur à discrétiser.
        intervalle (float): L'intervalle sur lequel la valeur est mesurée.
        nb_divisions (int): Le nombre de divisions pour discrétiser l'intervalle.
    Return:
        int: L'indice correspondant à la valeur discrétisée.
    """
    return min(nb_divisions - 1, max(0, int(val / intervalle * nb_divisions)))

def obtenir_etat_discret():
    """
    Donne l'état actuel discrétisé du jeu. L'état est composé de la position de la balle (x, y), de la vitesse de la balle (vx, vy), et de la position de la raquette de l'IA.
    Return:
        np.array: Un tableau contenant l'état discrétisé.
    """
    etat_x = discretiser(balle.x, largeur, 85)
    etat_y = discretiser(balle.y, hauteur, 85)  
    etat_vx = -1 if vitesse_balle_x < -2 else (1 if vitesse_balle_x > 2 else 0)  # -1, 0, 1 pour la vitesse x
    etat_vy = -1 if vitesse_balle_y < -2 else (1 if vitesse_balle_y > 2 else 0)  # -1, 0, 1 pour la vitesse y
    raquette_pos = discretiser(raquette2.y, hauteur, 55)  # 20-30 segments pour la raquette
    return np.array([etat_x, etat_y, etat_vx, etat_vy, raquette_pos], dtype=np.float32)

def choisir_action(state):
    """
    Choisit une action en fonction de l'état actuel en utilisant le réseau de neurones DQN.
    Args:
        state (np.array): L'état actuel du jeu sous forme de tableau.
    Return:
        int: L'indice de l'action choisie (0 = Monter, 1 = Descendre, 2 = Rester).
    """
    with torch.no_grad():
        state_tensor = torch.tensor(state).to(device)
        return policy_net(state_tensor).argmax().item()

def reinitialiser_jeu():
    """
    Réinitialise la position de la balle au centre de l'écran et réinitialise ses vitesses initiales.
    Cette fonction est appelée chaque fois qu'un joueur marque un point.
    """
    global balle, vitesse_balle_x, vitesse_balle_y
    balle = pygame.Rect(largeur // 2 - 15, hauteur // 2 - 15, 30, 30) # Repositionne la balle au centre
    vitesse_balle_x = -12 # Remet la vitesse à sa valeur initiale
    vitesse_balle_y = 6 * np.random.choice([-1, 1]) # Change la direction verticale aléatoirement

# Boucle principale
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Obtention de l'état actuel
    state = obtenir_etat_discret()

    # Contrôle de la raquette du joueur 1 (Humain)
    keys = pygame.key.get_pressed()
    if keys[pygame.K_z] and raquette1.top > 0: # Monter
        raquette1.y -= 10
    if keys[pygame.K_s] and raquette1.bottom < hauteur: # Descendre
        raquette1.y += 10

    # Action de l'IA pour la raquette du joueur 2
    action_idx = choisir_action(state)
    if action_idx == 0 and raquette2.top > 0: # Monter
        raquette2.y -= 10
    elif action_idx == 1 and raquette2.bottom < hauteur: # Descendre
        raquette2.y += 10

    # Mise à jour de la position de la balle
    balle.x += vitesse_balle_x
    balle.y += vitesse_balle_y

    # Collision avec le haut ou le bas de l'écran
    if balle.top <= 0 or balle.bottom >= hauteur:
        vitesse_balle_y = -vitesse_balle_y # Inverser la direction verticale

    # Collision avec les raquettes
    if balle.colliderect(raquette1):
        vitesse_balle_x = -vitesse_balle_x # Inverser la direction horizontale
        balle.left = raquette1.right # Correction de la position pour éviter les chevauchements
    if balle.colliderect(raquette2):
        vitesse_balle_x = -vitesse_balle_x # Inverser la direction horizontale
        balle.right = raquette2.left # Correction de la position pour éviter les chevauchements

    # Mise à jour des scores et réinitialisation si la balle sort de l'écran
    if balle.left <= 0: # Le joueur 2 marque un point
        score_joueur2 += 1
        reinitialiser_jeu()
    if balle.right >= largeur: # Le joueur 1 marque un point
        score_joueur1 += 1
        reinitialiser_jeu()

    # Dessin de l'écran de jeu
    fenetre.fill((0, 0, 0)) # Remplir l'écran avec la couleur noire
    pygame.draw.rect(fenetre, blanc, raquette1) # Dessiner la raquette du joueur 1
    pygame.draw.rect(fenetre, blanc, raquette2) # Dessiner la raquette de l'IA
    pygame.draw.ellipse(fenetre, blanc, balle) # Dessiner la balle

    # Affichage des scores
    score_text = font.render(f"Joueur 1: {score_joueur1}  Joueur 2: {score_joueur2}", True, blanc)
    fenetre.blit(score_text, (largeur // 2 - 100, 10))

    # Mise à jour de l'affichage
    pygame.display.flip()
    pygame.time.Clock().tick(60) # Limitation de la boucle à 60 images par seconde