#Jouer_vs_model

import pygame
import sys
import pickle
import random
import numpy as np

# Initialisation de pygame
pygame.init()

# Dimensions de la fenêtre du jeu
largeur, hauteur = 640, 480
fenetre = pygame.display.set_mode((largeur, hauteur))
pygame.display.set_caption("Pong - Jouer contre le Modèle")

# Couleurs
blanc = (255, 255, 255)

# Initialisation des raquettes et de la balle
raquette_joueur = pygame.Rect(50, hauteur // 2 - 70, 10, 140)  # Raquette du joueur
raquette_modele = pygame.Rect(largeur - 60, hauteur // 2 - 70, 10, 140)  # Raquette du modèle
balle = pygame.Rect(largeur // 2 - 15, hauteur // 2 - 15, 30, 30)  # Balle au centre

# Vitesses initiales de la balle
vitesse_balle_x = -12
vitesse_balle_y = 12 * random.choice([-1, 1])

# Variables pour les scores
score_joueur = 0
score_modele = 0

# Charger la Q-table
try:
    with open('q_table.pkl', 'rb') as f:
        q_table = pickle.load(f)
    print("Q-table chargée avec succès.")
except FileNotFoundError:
    print("Erreur : Aucune Q-table trouvée.")
    sys.exit()

# Actions possibles
actions = ["UP", "DOWN", "STAY"]

# Police de texte
police = pygame.font.Font(None, 36)

# Fonction pour discrétiser l'état
def discretiser(val, intervalle, nb_divisions):
    return min(nb_divisions - 1, max(0, int(val / intervalle * nb_divisions)))

def obtenir_etat_discret():
    etat_x = discretiser(balle.x, largeur, 20)
    etat_y = discretiser(balle.y, hauteur, 20)
    etat_vx = 0 if vitesse_balle_x < 0 else 1
    etat_vy = 0 if vitesse_balle_y < 0 else 1
    raquette_pos = discretiser(raquette_modele.y, hauteur, 20)
    return (etat_x, etat_y, etat_vx, etat_vy, raquette_pos)

# Fonction pour choisir une action (basée sur la Q-table)
def choisir_action(state):
    return actions[np.argmax(q_table[state])]

# Boucle principale du jeu
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Contrôle de la raquette du joueur
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w] and raquette_joueur.top > 0:
        raquette_joueur.y -= 10
    if keys[pygame.K_s] and raquette_joueur.bottom < hauteur:
        raquette_joueur.y += 10

    # Obtenir l'état actuel t
    etat = obtenir_etat_discret()

    # Le modèle choisit une action
    action = choisir_action(etat)

    # Exécution de l'action par la raquette du modèle
    if action == "UP" and raquette_modele.top > 0:
        raquette_modele.y -= 10
    elif action == "DOWN" and raquette_modele.bottom < hauteur:
        raquette_modele.y += 10

    # Déplacement de la balle
    balle.x += vitesse_balle_x
    balle.y += vitesse_balle_y

    # Rebond de la balle sur les bords
    if balle.top <= 0 or balle.bottom >= hauteur:
        vitesse_balle_y = -vitesse_balle_y

    # Collision avec la raquette du joueur
    if balle.colliderect(raquette_joueur):
        vitesse_balle_x = -vitesse_balle_x
        balle.left = raquette_joueur.right  # Déplacer la balle juste à droite de la raquette
        
    # Collision avec la raquette du modèle
    if balle.colliderect(raquette_modele):
        vitesse_balle_x = -vitesse_balle_x
        balle.right = raquette_modele.left

    # Balle sortie du jeu (perte de point)
    if balle.left <= 0:  # Perte pour le joueur
        score_modele += 1
        balle = pygame.Rect(largeur // 2 - 15, hauteur // 2 - 15, 30, 30)
        vitesse_balle_x = -12 * random.choice([-1, 1])
        vitesse_balle_y = 12 * random.choice([-1, 1])
    elif balle.right >= largeur:  # Perte pour le modèle
        score_joueur += 1
        balle = pygame.Rect(largeur // 2 - 15, hauteur // 2 - 15, 30, 30)
        vitesse_balle_x = 12 * random.choice([-1, 1])
        vitesse_balle_y = 12 * random.choice([-1, 1])

    # Effacer l'écran
    fenetre.fill((0, 0, 0))

    # Dessiner les raquettes et la balle
    pygame.draw.rect(fenetre, blanc, raquette_joueur)
    pygame.draw.rect(fenetre, blanc, raquette_modele)
    pygame.draw.ellipse(fenetre, blanc, balle)

    # Affichage des scores
    texte_joueur = police.render(str(score_joueur), True, blanc)
    texte_modele = police.render(str(score_modele), True, blanc)
    fenetre.blit(texte_joueur, (largeur // 4, 20))
    fenetre.blit(texte_modele, (3 * largeur // 4 - 20, 20))

    # Rafraîchir l'écran
    pygame.display.flip()

    # Limite de rafraîchissement
    pygame.time.Clock().tick(60)
