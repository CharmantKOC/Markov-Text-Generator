import re
from collections import defaultdict
import random
import numpy as np
import networkx as nx

class SimpleMarkovChain:
    """
    Implémentation simple d'une chaîne de Markov basée sur un dictionnaire
    de transitions construit manuellement.
    """
    
    def __init__(self):
        """
        Initialise la chaîne de Markov.
        """
        self.transitions = {}  # Dictionnaire des transitions
        self.states = []       # Liste des états possibles
        self.initial_states = []  # États initiaux possibles (début de phrase)
        self.transition_matrix = None  # Matrice de transition
        self.graph = nx.DiGraph()  # Graphe des transitions
        
    def tokenize_text(self, text):
        """
        Découpe le texte en tokens (mots et ponctuations).
        
        Args:
            text (str): Texte à tokeniser
            
        Returns:
            list: Liste de tokens
        """
        # Préserver la ponctuation comme tokens séparés
        text = re.sub(r'([.!?,;:])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.split()
    
    def set_transitions(self, transitions_dict):
        """
        Définit manuellement le dictionnaire de transitions.
        
        Args:
            transitions_dict (dict): Dictionnaire des transitions
                {état: {état_suivant: probabilité, ...}, ...}
        """
        self.transitions = transitions_dict
        self.states = list(transitions_dict.keys())
        
        # Identifier les états qui peuvent être des états initiaux (commencent par majuscule)
        self.initial_states = [state for state in self.states 
                            if state[0].isupper() or state == "."]
        
        # Construire la matrice de transition
        self._build_transition_matrix()
        
        # Construire le graphe
        self._build_graph()
    
    def _build_transition_matrix(self):
        """
        Construit la matrice de transition à partir du dictionnaire.
        """
        n = len(self.states)
        self.transition_matrix = np.zeros((n, n))
        
        # Indexation des états
        state_to_idx = {state: i for i, state in enumerate(self.states)}
        
        # Remplir la matrice
        for i, state in enumerate(self.states):
            if state in self.transitions:
                for next_state, prob in self.transitions[state].items():
                    if next_state in state_to_idx:
                        j = state_to_idx[next_state]
                        self.transition_matrix[i, j] = prob
    
    def _build_graph(self):
        """
        Construit le graphe des transitions.
        """
        self.graph = nx.DiGraph()
        
        # Ajouter les nœuds
        for state in self.states:
            self.graph.add_node(state)
        
        # Ajouter les arêtes avec leurs poids
        for state, transitions in self.transitions.items():
            for next_state, prob in transitions.items():
                if next_state in self.states:  # S'assurer que l'état est dans la liste des états
                    self.graph.add_edge(state, next_state, weight=prob, label=f"{prob:.2f}")
    
    def get_transition_matrix(self):
        """
        Retourne la matrice de transition avec les labels des états.
        
        Returns:
            tuple: (matrice, états)
        """
        return self.transition_matrix, self.states
    
    def get_graph(self):
        """
        Retourne le graphe des transitions.
        
        Returns:
            nx.DiGraph: Graphe des transitions
        """
        return self.graph
    
    def generate_text(self, length=20, start_state=None):
        """
        Génère du texte à partir du modèle.
        
        Args:
            length (int): Nombre de mots à générer
            start_state (str, optional): État initial. Si None, un état initial est choisi aléatoirement.
            
        Returns:
            str: Texte généré
        """
        if not self.states:
            return "Le modèle n'a pas été initialisé."
        
        # Choisir un état initial
        if start_state is None or start_state not in self.states:
            if self.initial_states:
                current_state = random.choice(self.initial_states)
            else:
                current_state = random.choice(self.states)
        else:
            current_state = start_state
        
        # Initialiser le texte généré
        result = [current_state]
        
        # Générer la séquence
        for _ in range(length - 1):
            if current_state in self.transitions and self.transitions[current_state]:
                # Obtenir les prochains états possibles et leurs probabilités
                next_states = list(self.transitions[current_state].keys())
                probabilities = list(self.transitions[current_state].values())
                
                # Choisir le prochain état selon les probabilités
                current_state = random.choices(next_states, weights=probabilities, k=1)[0]
                result.append(current_state)
                
                # Si on atteint un point final, on peut s'arrêter
                if current_state in [".", "!", "?"]:
                    break
            else:
                # Si pas de transition connue, choisir un état aléatoire
                current_state = random.choice(self.states)
                result.append(current_state)
        
        # Nettoyer le résultat (ajuster les espaces autour de la ponctuation)
        text = " ".join(result)
        text = re.sub(r'\s+([.!?,;:])', r'\1', text)
        text = re.sub(r'([.!?,;:])\s+([A-Z])', r'\1 \2', text)
        
        return text
    
    @staticmethod
    def build_transitions_from_text(text, order=1):
        """
        Construit un dictionnaire de transitions à partir d'un texte.
        Cette méthode est utile pour créer le dictionnaire manuellement, 
        mais on peut aussi le modifier ensuite.
        
        Args:
            text (str): Texte d'entrée
            order (int): Ordre de la chaîne de Markov
            
        Returns:
            dict: Dictionnaire des transitions
        """
        # Tokeniser le texte
        text = re.sub(r'([.!?,;:])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = text.split()
        
        if len(tokens) <= order:
            return {}
        
        # Construire les transitions
        transitions = defaultdict(lambda: defaultdict(int))
        
        # Compter les occurrences
        for i in range(len(tokens) - order):
            current = tokens[i]
            next_token = tokens[i + order]
            transitions[current][next_token] += 1
        
        # Convertir les compteurs en probabilités
        result = {}
        for current, nexts in transitions.items():
            total = sum(nexts.values())
            result[current] = {next_token: count/total for next_token, count in nexts.items()}
        
        return result