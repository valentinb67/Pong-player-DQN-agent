import pygame
import sys
import random
import numpy as np
from collections import defaultdict

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

# Variables pour les scores
score1 = 0
score2 = 0

# Variables pour les compteurs
compteur_global = 0  # Compteur du nombre total de touches
compteur_session = 0  # Compteur de touches pour la session actuelle

# Statistiques d'entraînement
epsilon_hist = [] 
reward_cumule_hist = []
reward_cumule_episode = 0 
nombre_episodes = 0 

# Police de texte
police = pygame.font.Font(None, 36)

# Q-Learning
actions = ["UP", "DOWN", "STAY"]
q_table = defaultdict(lambda: np.zeros(len(actions)))
alpha = 0.7  
gamma = 0.7  
epsilon = 0.9  

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
    vitesse_balle_y = 12 * random.choice([-1, 1])

# Boucle principale du jeu
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
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
        compteur_global += 1  # Incrément du compteur global
        compteur_session += 1  # Incrément du compteur de session
        
    # Collision avec la raquette 2
    if balle.colliderect(raquette2):
        vitesse_balle_x = -vitesse_balle_x
        balle.right = raquette2.left # Déplacer la balle juste à gauche de la raquette
        reward = 1  # Récompense positive quand la balle est renvoyée
        reward_cumule_episode += reward  # Ajouter à la récompense cumulée
        compteur_global += 1  # Incrément du compteur global
        compteur_session += 1  # Incrément du compteur de session

    # Balle sortie du jeu (perte de point)
    if balle.left <= 0:  # Si la balle sort du côté du joueur
        score2 += 1
        reward = -10
        reward_cumule_episode += reward
        compteur_session = 0  # Réinitialiser le compteur de session
        nombre_episodes += 1  # Fin de l'épisode
        reward_cumule_hist.append(reward_cumule_episode)  # Stocker la récompense cumulée
        epsilon_hist.append(epsilon)  # Stocker epsilon
        reward_cumule_episode = 0  # Réinitialiser les récompenses cumulées pour le prochain épisode
        reinitialiser_jeu()  # Réinitialiser l'environnement pour le nouvel épisode

    if balle.right >= largeur:  # Si la balle sort du côté de l'ordinateur
        score1 += 1
        reward = -10  # Récompense positive pour avoir marqué un point
        reward_cumule_episode += reward
        compteur_session = 0  # Réinitialiser le compteur de session
        nombre_episodes += 1  # Fin de l'épisode
        reward_cumule_hist.append(reward_cumule_episode)  # Stocker la récompense cumulée
        epsilon_hist.append(epsilon)  # Stocker epsilon
        reward_cumule_episode = 0  # Réinitialiser les récompenses cumulées
        reinitialiser_jeu()  # Réinitialiser l'environnement

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

    # Affichage des compteurs
    texte_compteur_global = police.render(f"Touches globales: {compteur_global}", True, blanc)
    texte_compteur_session = police.render(f"Touches session: {compteur_session}", True, blanc)
    fenetre.blit(texte_compteur_global, (20, hauteur - 60))
    fenetre.blit(texte_compteur_session, (20, hauteur - 30))

    # Affichage des statistiques d'entraînement
    texte_episodes = police.render(f"Épisodes: {nombre_episodes}", True, blanc)
    texte_epsilon = police.render(f"Epsilon: {epsilon:.2f}", True, blanc)
    texte_reward = police.render(f"Récompense: {reward_cumule_episode}", True, blanc)
    fenetre.blit(texte_episodes, (largeur - 150, hauteur - 60))
    fenetre.blit(texte_epsilon, (largeur - 150, hauteur - 40))
    fenetre.blit(texte_reward, (largeur - 150, hauteur - 20))

    # Rafraîchir l'écran
    pygame.display.flip()

    # Limite de rafraîchissement
    pygame.time.Clock().tick(60)