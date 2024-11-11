import pygame
import sys
import random
import numpy as np
import cv2
from collections import defaultdict
import csv
import time

# Initialisation de pygame
pygame.init()

# Dimensions de la fenêtre du jeu
largeur, hauteur = 640, 480
fenetre = pygame.display.set_mode((largeur, hauteur))
pygame.display.set_caption("Pong")

# Couleurs
blanc = (255, 255, 255)

# Initialisation des raquettes et de la balle
raquette1 = pygame.Rect(50, hauteur // 2 - 70, 10, 140)  # Raquette du joueur 1 (Gauche)
raquette2 = pygame.Rect(largeur - 60, hauteur // 2 - 70, 10, 140)  # Raquette du joueur 2 (l'ordinateur)
balle = pygame.Rect(largeur // 2 - 15, hauteur // 2 - 15, 30, 30)  # Balle au centre

# Vitesses initiales de la balle
vitesse_balle_x = -12
vitesse_balle_y = 6 * random.choice([-1, 1])

# Variables pour les scores
score1 = 0
score2 = 0

# Police de texte
police = pygame.font.Font(None, 36)

# Boucle principale du jeu
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Mouvement des raquettes
    touches = pygame.key.get_pressed()
    if touches[pygame.K_UP] and raquette2.top > 0:
        raquette2.y -= 10
    if touches[pygame.K_DOWN] and raquette2.bottom < hauteur:
        raquette2.y += 10
        
    # Tracker
    raquette1.y = balle.y
    
    # Déplacement de la balle
    balle.x += vitesse_balle_x
    balle.y += vitesse_balle_y

    # Rebond de la balle sur les bords
    if balle.top <= 0 or balle.bottom >= hauteur:
        vitesse_balle_y = -vitesse_balle_y

    # Collision avec les raquettes
    if balle.colliderect(raquette1) or balle.colliderect(raquette2):
        vitesse_balle_x = -vitesse_balle_x

    # Balle sortie du jeu (marquer un point)
    if balle.left <= 0: # J2 marque un point
        score2 += 1
        balle = pygame.Rect(largeur // 2 - 15, hauteur // 2 - 15, 30, 30)
        vitesse_balle_x = 12

    if balle.right >= largeur: # J1 marque un point
        score1 += 1
        balle = pygame.Rect(largeur // 2 - 15, hauteur // 2 - 15, 30, 30)
        vitesse_balle_x = -12

    # Effacement de l'écran
    fenetre.fill((0, 0, 0))

    # Dessin des raquettes et de la balle
    pygame.draw.rect(fenetre, blanc, raquette1)
    pygame.draw.rect(fenetre, blanc, raquette2)
    pygame.draw.ellipse(fenetre, blanc, balle)

    # Rafraîchissement de l'écran
    pygame.display.flip()

    # Limite de vitesse de rafraîchissement
    pygame.time.Clock().tick(60)