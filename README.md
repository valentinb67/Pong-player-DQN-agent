# Pong-player-DQN-agent

As part of the unsupervised learning course in the second year of the Master Data science for business and economics program, this project aims to train a DQN Agent to play in a Pong environment. We'll be looking at a number of approaches, including training with a Q-Learning Agent, before moving on to a DQN Agent which, thanks to its NN architecture, will be able to converge more quickly in a 640*480 environment. We'll also be playing with the DQN's hyperparameters, for which we'll exploit GPU resources from the pytorch package using cuda.

# Les scripts
## Les Scripts d'Entrainement:

**1-pong_Qlearning_Discret.py:** Permet d'entrainer le modèle Q-learning en environnement discret.\
Permet d'obtenir en sortie :
- Les données d'entrainement "LearningData/1_q_learning_log.csv"
- La vidéo "Records/1_q_table_record.avi" (non disponible sur github en raison du poids de la video)

**1.2-pong_Qlearning_rdm.py:** Permet d'entrainer le modèle Q-learning en environnement discret avec récompenses aléatoires pour chacune des actions.\
Permet d'obtenir en sortie :
- Les données d'entrainement "LearningData/1.2_q_learning_rdm_log.csv"
- La vidéo "Records/1.2_q_record_rdm.avi" (non disponible sur github en raison du poids de la video)

**1.5-pong_Qlearning_Continu.py:** Permet d'entrainer le modèle Q-learning en environnement continu.\
Permet d'obtenir en sortie :
- Les données d'entrainement "LearningData/1.5_q_learning_continuous_log.csv"

**2-pong_DQN_Discret.py:** Permet d'entrainer le modèle Deep Q-Network en environnement discret.\
Permet d'obtenir en sortie :
- Les données d'entrainement "LearningData/2_pong_dqn_discret_training_log.csv"
- Le modèle entrainé "ModelsPTH/2_dqn_discret.pth"

**2.5-pong_DQN_Continu.py:** Permet d'entrainer le modèle Deep Q-Network en environnement continu.\
Permet d'obtenir en sortie :
- Les données d'entrainement "LearningData/2.5_pong_dqn_continuous_training_log.csv"
- Le modèle entrainé "ModelsPTH/2.5_dqn_continuous.pth"

**3-pong_DDQN_Discret.py:** Permet d'entrainer le modèle Double Deep Q-Network en environnement discret.\
Permet d'obtenir en sortie : 
- Les données d'entrainement "LearningData/3_pong_double_dqn_discret_training_log.csv"
- Le modèle entrainé "ModelsPTH/3_Ddqn_discret.pth"

**3.5-pong_DDQN_Continu.py:** Permet d'entrainer le modèle Double Deep Q-Network en environnement continu.\
Permet d'obtenir en sortie :
- Les données d'entrainement "LearningData/3.5_pong_double_dqn_continuous_training_log.csv"
- Le modèle entrainé "ModelsPTH/3.5_Ddqn_continuous.pth"
- La vidéo "Records/3.5_Ddqn_record.avi" (non disponible sur github en raison du poids de la video)

## Analyse de Données d'Apprentissage et Création de Supports Annexe:
**1-Q_learning_analysis.py:** Analyse des données d'apprentissage pour les modèles Q-learning.\
**2-DQN_and_DDQN_discret_analysis.py:** Analyse des données d'apprentissage pour les modèles Deep Q-Network et Double Deep Q-Network.\
**EpsilonDecay_MarginEffect_analysis.py:** Met en évidence l'effet marginal de *Epsilon Decay* sur la vitesse de transition entre *epsilon* et *epsilon_min*.\
**Space_size.py** Met en évidence la croissance exponentielle du nombre d'état dans l'espace des états pour une augmentation du nombre de pixel.

## Confrontation aux modèles
Les scripts suivants permettent de se confronter aux modèles DDQN afin de tester leur performances hors indicateurs (ou pour simplement passer un bon moment :D).\
**#Play vs DDQN discret.py:** Permet de jouer face au DDQN discret.
**#Play vs DDQN continu.py:** Permet de jouer face au DDQN continu.

# Fonctions Scripts d'Entrainement Modèles

**discretiser(*val*, *intervalle*, *nb_divisions*):**
Permet de convertir une valeur continue en une valeur discrète, en la divisant en un nombre spécifié de divisions (*nb_divisions*). Elle est utilisée pour réduire l'espace d'état à des valeurs discrètes, ce qui est essentiel pour l'algorithme de Q-learning qui fonctionne avec des états discrets.

**obtenir_etat_discret():**
Génère un état discret du jeu en utilisant la fonction discretiser. Elle récupère la position de la balle, sa direction, ainsi que la position de la raquette de l’ordinateur, puis retourne ces valeurs sous forme de tuple. Cela permet de créer un état compact et gérable pour l’apprentissage par renforcement.

**choisir_action(*state*):**
Implémente la stratégie epsilon-greedy pour choisir une action basée sur l’état actuel. Elle retourne une action aléatoire avec une probabilité epsilon pour encourager l'exploration, ou l’action avec la meilleure estimation de récompense selon la q_table pour l’exploitation.

**mise_a_jour_q_table(*etat*, *action*, *reward*, *etat_suivant*):**
Met à jour la Q-table en appliquant l'algorithme de Q-learning. Elle utilise l'état actuel, l'action prise, la récompense obtenue, et l'état suivant pour ajuster la valeur associée dans la Q-table. Elle retourne aussi des valeurs comme l’erreur de temporal difference (*td_error*), la valeur attendue (*true_value*), la valeur estimée (*estimate_value*), et la perte (*loss*) pour enregistrement.

**reinitialiser_jeu():**
Réinitialise les positions de la balle et sa vitesse pour le début d’un nouvel épisode. Elle est appelée chaque fois qu'un épisode se termine, notamment quand la balle sort du jeu.

**obtenir_etat_continu():**
Crée une représentation continue de l’état en normalisant les positions de la balle et de la raquette, ainsi que les vitesses de la balle. Elle retourne un tuple contenant ces valeurs continues, qui peuvent être utilisées comme état pour la Q-table. Cette version continue de l'état offre une alternative à la version discrète précédemment utilisée, permettant potentiellement une approche plus fine dans l’apprentissage.

**stocker_transition(*state*, *action*, *reward*, *next_state*, *done*):**
Enregistre les transitions (état, action, récompense, état suivant, et fin d’épisode) dans une mémoire tampon (*deque*). Cette mémoire est ensuite utilisée pour former des mini-lots pour l'entraînement du réseau neuronal, permettant ainsi un apprentissage par mini-lots avec répétition d'expériences passées (*replay memory*), ce qui stabilise l'apprentissage du modèle DQN.

**entrainer_dqn():**
Effectue l'entraînement du modèle DQN à partir d'un mini-lot de transitions échantillonnées dans la mémoire. Elle calcule les valeurs Q estimées pour chaque état-action, déduit les valeurs cibles en prenant la meilleure estimation pour l'état suivant (cible DQN). Elle calcule ensuite la perte et ajuste le réseau de politique (*policy_net*). Elle retourne également la perte et les valeurs True Value, Value Estimate, et TD Error pour un suivi de la progression d’apprentissage.

**entrainer_ddqn():**
Contrairement à la fonction **entrainer_dqn**, l'algorithme Double DQN est appliqué et sélectionne l’action optimale dans l’état suivant (*next_actions*) en utilisant le réseau de politique (*policy_net*) puis évalue cette action avec le réseau cible (*target_net*). Cette séparation permet de réduire le biais de surestimation des valeurs Q, ce qui stabilise l’apprentissage.

# Variables

## Variables de configuration et d'initialisation
*largeur*, *hauteur* : Dimensions de la fenêtre du jeu.\
*fenetre* : Surface de la fenêtre de jeu créée avec Pygame.\
*blanc* : Couleur blanche utilisée pour dessiner les objets.\
*font*: Police de texte pour l'affichage des scores.\
*device* : Périphérique d'exécution pour PyTorch, configuré sur CUDA si disponible, sinon CPU.

## Variables pour les objets du jeu
*raquette1*, *raquette2* : Raquettes des joueurs, initialisées en tant que rectangles Pygame.\
*balle* : Rectangle représentant la balle de jeu.\
*vitesse_balle_x*, *vitesse_balle_y* : Vitesse initiale de la balle en x et y. Ces valeurs changent au cours du jeu en fonction des collisions.

## Variables de scores
*score1*, *score2* : Scores des joueurs pour les scripts avec Q-learning.\
*score_joueur1*, *score_joueur2* : Scores des joueurs pour les scripts utilisant DQN et Double DQN.

## Hyperparamètres d'apprentissage
*alpha* : Taux d'apprentissage pour Q-learning ou DQN.\
*gamma* : Facteur d'actualisation (discount factor) pour l'apprentissage par renforcement.\
*epsilon* : Taux d'exploration initial pour les stratégies epsilon-greedy.\
*epsilon_decay* : Facteur de décroissance de epsilon à la fin de chaque épisode.\
*epsilon_min* : Valeur minimale de epsilon.\
*batch_size* : Taille du lot pour l'apprentissage DQN/Double DQN.\
*memory_size* : Taille maximale de la mémoire de replay.\
*target_update* : Fréquence (en nombre d'itérations) de la mise à jour du réseau cible dans DQN/Double DQN.\
*nb_actions* : Nombre d'actions disponibles (3 : "UP", "DOWN", "STAY").

## Variables pour la gestion des épisodes et de la mémoire
*max_episodes* : Nombre maximal d'épisodes pour l'apprentissage.\
*episode_count* : Compteur d'épisodes pour suivre le nombre d'épisodes joués.\
*memory* : Mémoire de replay utilisée pour stocker les transitions d'états, représentée par une deque.\
*frames* : Compteur de frames, utilisé pour déterminer quand mettre à jour le réseau cible dans DQN/Double DQN.

## Variables de récompenses et de suivi d'épisode (Q et TD error)
*reward* : Récompense immédiate obtenue dans un état donné.\
*compteur_global*, *compteur_session* : Nombre de fois où l'agent renvoie la balle durant la partie, par épisode (session).\
*episode_reward* : Récompense totale accumulée pendant un épisode.\
*episode_duration* : Durée totale d'un épisode, mesurée en secondes.\
*done* : Booléen indiquant la fin d'un épisode lorsque la balle sort des limites de jeu.\
*true_values*, *value_estimates* : Valeurs réelles et estimées des Q-values pour le suivi des performances d'apprentissage.\
*td_errors* : Erreurs de différence temporelle (TD error) entre les valeurs réelles et estimées, utilisées pour le suivi et le calcul de la perte.\
*episode_loss* : Somme des pertes pour un épisode dans les DQN (déduite du TD error).

## Variables pour l'enregistrement des données et les fichiers CSV
*csv_file* : Fichier CSV pour enregistrer les statistiques d'apprentissage.\
*csv_writer* : Objet permettant d'écrire des lignes dans le fichier CSV.
