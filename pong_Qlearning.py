import pygame
import sys
import random
import numpy as np
import cv2  # Pour l'enregistrement de la vidéo
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
raquette1 = pygame.Rect(50, hauteur // 2 - 70, 10, 140)  # Raquette du joueur
raquette2 = pygame.Rect(largeur - 60, hauteur // 2 - 70, 10, 140)  # Raquette de l'ordinateur
balle = pygame.Rect(largeur // 2 - 15, hauteur // 2 - 15, 30, 30)  # Balle au centre

# Vitesses initiales de la balle
vitesse_balle_x = -12
vitesse_balle_y = 6 * random.choice([-1, 1])

# Variables pour les scores
score1 = 0
score2 = 0

# Variables pour les compteurs
compteur_global = 0  # Compteur du nombre total de touches
compteur_session = 0  # Compteur de touches pour la session actuelle

# Statistiques d'entraînement
episode_count = 0
max_episodes = 500

# Hyperparamètres du Q-Learning
actions = ["UP", "DOWN", "STAY"]
q_table = defaultdict(lambda: np.zeros(len(actions)))
alpha = 0.7  
gamma = 0.7  

# Paramètres pour l'exploration epsilon-greedy
epsilon = 0.9  # Valeur initiale de epsilon
epsilon_decay = 0.975  # Facteur de décroissance de epsilon
epsilon_min = 0.1  # Valeur minimale pour epsilon

# CSV File for logging
csv_file = open('q_learning_log.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Episode', 'Epsilon', 'Touches Session', 'Touches Globales', 'Reward'])

# Police de texte
police = pygame.font.Font(None, 36)

# Initialisation pour l'enregistrement vidéo
fps = 60
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('1_q_table_record.avi', fourcc, fps, (largeur, hauteur))

# Fonction pour discrétiser l'état
def discretiser(val, intervalle, nb_divisions):
    return min(nb_divisions - 1, max(0, int(val / intervalle * nb_divisions)))

def obtenir_etat_discret():
    etat_x = discretiser(balle.x, largeur, 20)
    etat_y = discretiser(balle.y, hauteur, 20)
    etat_vx = 0 if vitesse_balle_x < 0 else 1
    etat_vy = 0 if vitesse_balle_y < 0 else 1
    raquette_pos = discretiser(raquette2.y, hauteur, 20)
    return (etat_x, etat_y, etat_vx, etat_vy, raquette_pos)

# Fonction pour choisir une action (epsilon-greedy)
def choisir_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    else:
        return actions[np.argmax(q_table[state])]

# Fonction de mise à jour Q-Learning
def mise_a_jour_q_table(etat, action, reward, etat_suivant):
    action_idx = actions.index(action)
    meilleure_action_suivante = np.max(q_table[etat_suivant])
    q_table[etat][action_idx] = q_table[etat][action_idx] + alpha * (reward + gamma * meilleure_action_suivante - q_table[etat][action_idx])

# Fonction pour réinitialiser l'environnement après chaque épisode
def reinitialiser_jeu():
    global balle, vitesse_balle_x, vitesse_balle_y
    balle = pygame.Rect(largeur // 2 - 15, hauteur // 2 - 15, 30, 30)
    vitesse_balle_x = -12
    vitesse_balle_y = 6 * random.choice([-1, 1])

# Boucle principale du jeu
while episode_count < max_episodes:  # Condition de fin basée sur max_episodes
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            csv_file.close()
            print("Q-table enregistrée avec succès.")
            video_writer.release()
            pygame.quit()
            sys.exit()

    # Initialiser la récompense à 0 au début de chaque boucle
    reward = 0

    # Mouvement de la raquette du joueur 1 pour tracker la balle
    raquette1.y = balle.y

    # Obtenir l'état actuel t
    etat = obtenir_etat_discret()

    # L'ordinateur choisit une action
    action = choisir_action(etat)

    # Exécution de l'action
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

    # Collision avec la raquette 1
    if balle.colliderect(raquette1):
        vitesse_balle_x = -vitesse_balle_x
        balle.left = raquette1.right  # Déplacer la balle juste à droite de la raquette
        
    # Collision avec la raquette 2
    if balle.colliderect(raquette2):
        vitesse_balle_x = -vitesse_balle_x
        balle.right = raquette2.left
        reward = 1
        compteur_global += 1
        compteur_session += 1

    # Balle sortie du jeu (perte de point)
    if balle.left <= 0:  # Le joueur ordinateur marque un point
        score2 += 1
        reward = -10
        episode_count += 1  # Fin de l'épisode
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        reinitialiser_jeu()
        compteur_session = 0
        continue  # Passer au prochain épisode

    elif balle.right >= largeur:  # Le joueur humain marque un point
        score1 += 1
        reward = -10
        episode_count += 1
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        reinitialiser_jeu()
        compteur_session = 0
        continue

    # Obtenir le nouvel état
    etat_suivant = obtenir_etat_discret()

    # Mise à jour de la Q-Table
    mise_a_jour_q_table(etat, action, reward, etat_suivant)

    # Effacer l'écran
    fenetre.fill((0, 0, 0))

    # Dessiner les raquettes et la balle
    pygame.draw.rect(fenetre, blanc, raquette1)
    pygame.draw.rect(fenetre, blanc, raquette2)
    pygame.draw.ellipse(fenetre, blanc, balle)

    # Affichage des scores
    texte1 = police.render(str(score1), True, blanc)
    texte2 = police.render(str(score2), True, blanc)
    fenetre.blit(texte1, (largeur // 4, 20))
    fenetre.blit(texte2, (3 * largeur // 4 - 20, 20))

    # Affichage de l'épisode et de la valeur epsilon en cours
    episode_text = police.render(f"Episode: {episode_count}  Epsilon: {epsilon:.3f}", True, blanc)
    fenetre.blit(episode_text, (10, hauteur - 40))

    # Rafraîchir l'écran
    pygame.display.flip()

    # Capture de l'écran pour l'enregistrement vidéo
    frame = pygame.surfarray.array3d(pygame.display.get_surface())
    frame = cv2.transpose(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video_writer.write(frame)

    # Limite de rafraîchissement
    pygame.time.Clock().tick(fps)

# Fermer le fichier CSV après la fin de l'entraînement
csv_file.close()
video_writer.release()
pygame.quit()
