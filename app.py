import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components
import json
from Markov import SimpleMarkovChain

# Modification de la classe SimpleMarkovChain pour supporter le nombre de phrases
# Note: Cette modification assume que la classe peut être étendue ou existe déjà avec cette méthode
class EnhancedMarkovChain(SimpleMarkovChain):
    def generate_text(self, length=20, start_state=None, num_sentences=1):
        """
        Génère du texte en spécifiant le nombre de phrases à générer
        
        Args:
            length: Nombre maximum de mots
            start_state: État initial (mot)
            num_sentences: Nombre de phrases à générer
        
        Returns:
            Texte généré
        """
        # Si un seul nombre de phrases est demandé, utiliser la méthode standard
        if num_sentences <= 1:
            return super().generate_text(length, start_state)
            
        # Pour plusieurs phrases, toutes avec le même mot de départ si spécifié
        sentences = []
        for _ in range(num_sentences):
            # Utiliser le même état initial pour chaque phrase si spécifié
            new_sentence = super().generate_text(length, start_state)
            sentences.append(new_sentence)
            
        # Joindre les phrases avec un saut de ligne entre chaque
        return "\n".join(sentences)

# Configuration de la page
st.set_page_config(
    page_title="Générateur par Chaîne de Markov",
    page_icon="📝",
    layout="wide"
)

# Titre et description concise
st.title("Générateur de texte par chaînes de Markov")
st.markdown("""
Application permettant de créer, visualiser et utiliser des chaînes de Markov pour la génération de texte.
""")

# Initialiser le modèle et les variables de session
if 'markov_chain' not in st.session_state:
    st.session_state.markov_chain = EnhancedMarkovChain()

if 'transitions' not in st.session_state:
    st.session_state.transitions = {}

if 'input_text' not in st.session_state:
    st.session_state.input_text = "Le chat noir dort. Le chat blanc joue. Le chien dort aussi."

# Créer les onglets avec noms concis
tab1, tab2, tab3, tab4 = st.tabs(["Données d'entrée", "Configuration", "Visualisation", "Génération"])

# Onglet 1: Texte d'entrée
with tab1:
    st.header("Données d'entrée")
    
    input_text = st.text_area(
        "Texte d'entraînement",
        value=st.session_state.input_text,
        height=200
    )
    
    st.session_state.input_text = input_text
    
    if st.button("Construire le modèle"):
        transitions = SimpleMarkovChain.build_transitions_from_text(input_text)
        st.session_state.transitions = transitions
        st.success(f"Modèle créé avec {len(transitions)} états")
        
        with st.expander("Aperçu du modèle"):
            st.json(transitions)

# Onglet 2: Dictionnaire de transitions
with tab2:
    st.header("Configuration du modèle")
    
    st.caption("""
    Chaque ligne représente un état et ses transitions possibles.
    Format: `état -> état_suivant:probabilité, état_suivant:probabilité, ...`
    """)
    
    # Convertir le dictionnaire en texte pour l'édition
    transitions_text = ""
    for state, next_states in st.session_state.transitions.items():
        transitions_line = f"{state} -> "
        transitions_items = [f"{next_state}:{prob:.2f}" for next_state, prob in next_states.items()]
        transitions_line += ", ".join(transitions_items)
        transitions_text += transitions_line + "\n"
    
    edited_transitions = st.text_area(
        "Matrice de transition",
        value=transitions_text,
        height=300
    )
    
    if st.button("Appliquer les modifications"):
        # Parser le texte en dictionnaire
        transitions = {}
        lines = edited_transitions.strip().split("\n")
        
        for line in lines:
            if "->" in line:
                state, transitions_str = line.split("->")
                state = state.strip()
                transitions[state] = {}
                
                if transitions_str.strip():
                    transitions_items = transitions_str.split(",")
                    for item in transitions_items:
                        if ":" in item:
                            next_state, prob_str = item.split(":")
                            next_state = next_state.strip()
                            try:
                                prob = float(prob_str.strip())
                                transitions[state][next_state] = prob
                            except ValueError:
                                st.error(f"Erreur de format: {prob_str}")
        
        # Vérifier que les probabilités somment à 1 pour chaque état
        valid = True
        for state, next_states in transitions.items():
            total_prob = sum(next_states.values())
            if abs(total_prob - 1.0) > 0.01:  # Tolérance pour les erreurs d'arrondi
                st.warning(f"État '{state}': somme des probabilités = {total_prob:.2f} (≠ 1.0)")
                valid = False
        
        if valid:
            st.session_state.transitions = transitions
            markov_chain = EnhancedMarkovChain()
            markov_chain.set_transitions(transitions)
            st.session_state.markov_chain = markov_chain
            st.success("Modèle mis à jour")
        else:
            st.error("Veuillez corriger les erreurs dans la matrice de transition")

# Onglet 3: Visualisation
with tab3:
    st.header("Visualisation du modèle")
    
    if st.session_state.transitions:
        # Mettre à jour le modèle si nécessaire
        if not hasattr(st.session_state.markov_chain, 'transitions') or st.session_state.markov_chain.transitions != st.session_state.transitions:
            markov_chain = EnhancedMarkovChain()
            markov_chain.set_transitions(st.session_state.transitions)
            st.session_state.markov_chain = markov_chain
        
        # Créer les sous-onglets de visualisation
        vis_tab1, vis_tab2 = st.tabs(["Graphe", "Matrice"])
        
        # Sous-onglet 1: Graphe des transitions
        with vis_tab1:
            st.subheader("Graphe des transitions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                vis_type = st.radio(
                    "Mode d'affichage",
                    ["Statique", "Interactif"]
                )
            
            with col2:
                # Ne pas limiter le nombre de nœuds par défaut
                display_all_nodes = st.checkbox("Afficher tous les nœuds", value=True)
                
                if not display_all_nodes:
                    node_limit = st.slider(
                        "Nombre max. de nœuds",
                        min_value=5,
                        max_value=50,
                        value=20
                    )
                else:
                    node_limit = len(st.session_state.transitions)
            
            # Obtenir le graphe
            graph = st.session_state.markov_chain.get_graph()
            
            # Limiter le nombre de nœuds si demandé et nécessaire
            if not display_all_nodes and len(graph.nodes()) > node_limit:
                st.info(f"Affichage limité à {node_limit} nœuds sur {len(graph.nodes())} au total.")
                nodes = list(graph.nodes())[:node_limit]
                graph = graph.subgraph(nodes)
            
            if vis_type == "Statique":
                # Créer le graphique avec matplotlib
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Positions des nœuds
                pos = nx.spring_layout(graph, seed=42)
                
                # Dessiner les nœuds
                nx.draw_networkx_nodes(graph, pos, node_size=700, node_color='lightblue', alpha=0.8, ax=ax)
                
                # Dessiner les arêtes avec des flèches
                edge_weights = [graph[u][v]['weight'] * 3 for u, v in graph.edges()]
                nx.draw_networkx_edges(graph, pos, width=edge_weights, alpha=0.7, 
                                      arrows=True, arrowsize=15, edge_color='gray', ax=ax)
                
                # Ajouter les étiquettes
                nx.draw_networkx_labels(graph, pos, font_size=10, ax=ax)
                
                # Ajouter les étiquettes des arêtes (probabilités)
                edge_labels = {(u, v): f"{graph[u][v]['weight']:.2f}" for u, v in graph.edges()}
                nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, ax=ax)
                
                # Configurer l'apparence
                plt.title("Graphe de transition")
                plt.axis('off')
                
                # Afficher le graphique
                st.pyplot(fig)
            else:
                # Visualisation interactive avec Pyvis
                net = Network(notebook=True, height="600px", width="100%", directed=True)
                
                # Ajouter les nœuds
                for node in graph.nodes():
                    net.add_node(node, label=node, title=node)
                
                # Ajouter les arêtes avec des largeurs proportionnelles aux poids
                for u, v, data in graph.edges(data=True):
                    weight = data.get('weight', 0.5)
                    width = 1 + 5 * weight  # Largeur proportionnelle à la probabilité
                    net.add_edge(u, v, value=weight, title=f"P: {weight:.2f}", width=width)
                
                # Configurer la physique du réseau
                net.toggle_physics(True)
                net.barnes_hut(gravity=-5000, central_gravity=0.3, spring_length=100)
                
                # Sauvegarder et afficher
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                    net.save_graph(tmp.name)
                    components.html(open(tmp.name, 'r', encoding='utf-8').read(), height=600)
        
        # Sous-onglet 2: Matrice de transition
        with vis_tab2:
            st.subheader("Matrice de transition")
            
            # Obtenir la matrice de transition
            matrix, states = st.session_state.markov_chain.get_transition_matrix()
            
            # Créer un DataFrame pour afficher la matrice
            df = pd.DataFrame(matrix, index=states, columns=states)
            
            # Afficher la matrice
            st.caption("Probabilités de transition entre états (ligne → colonne)")
            st.dataframe(df)
            
            # Visualiser la matrice sous forme de heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            plt.title("Matrice de transition")
            
            # Créer la heatmap
            im = ax.imshow(matrix, cmap='Blues')
            
            # Ajouter les annotations
            for i in range(len(states)):
                for j in range(len(states)):
                    if matrix[i, j] > 0:
                        text = ax.text(j, i, f"{matrix[i, j]:.2f}",
                                      ha="center", va="center", color="black" if matrix[i, j] < 0.5 else "white")
            
            # Ajouter les étiquettes des axes
            ax.set_xticks(np.arange(len(states)))
            ax.set_yticks(np.arange(len(states)))
            ax.set_xticklabels(states)
            ax.set_yticklabels(states)
            
            # Rotation des étiquettes de l'axe x pour une meilleure lisibilité
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Ajouter une barre de couleur
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel("Probabilité", rotation=-90, va="bottom")
            
            # Afficher la heatmap
            st.pyplot(fig)
    else:
        st.info("Veuillez d'abord configurer le modèle dans l'onglet Configuration.")

# Onglet 4: Génération
with tab4:
    st.header("Génération de texte")
    
    if st.session_state.transitions:
        # Options de génération
        col1, col2 = st.columns(2)
        
        with col1:
            # Sélectionner l'état initial
            initial_states = list(st.session_state.transitions.keys())
            initial_state = st.selectbox(
                "État initial",
                ["Aléatoire"] + initial_states
            )
            
            if initial_state == "Aléatoire":
                initial_state = None
                
            st.caption("L'état initial sera utilisé pour toutes les phrases générées.")
        
        with col2:
            # Options de génération
            length = st.slider(
                "Nombre de mots",
                min_value=5,
                max_value=100,
                value=20
            )
            
            sentences = st.number_input(
                "Nombre de phrases",
                min_value=1,
                max_value=20,
                value=3,
                step=1
            )
        
        # Bouton pour générer le texte
        if st.button("Générer"):
            try:
                # Générer le texte
                generated_text = st.session_state.markov_chain.generate_text(
                    length=length,
                    start_state=initial_state,
                    num_sentences=sentences
                )
                
                # Afficher le texte généré
                st.subheader("Résultat")
                
                # Séparer les phrases pour un meilleur affichage
                if "\n" in generated_text:
                    for i, sentence in enumerate(generated_text.split("\n")):
                        st.text(f"Phrase {i+1}: {sentence}")
                else:
                    st.text(generated_text)
                
                # Ajouter à l'historique des générations
                if 'generation_history' not in st.session_state:
                    st.session_state.generation_history = []
                
                st.session_state.generation_history.append({
                    "texte": generated_text,
                    "état_initial": initial_state if initial_state else "Aléatoire",
                    "longueur": length,
                    "phrases": sentences
                })
            except Exception as e:
                st.error(f"Erreur: {str(e)}")
        
        # Afficher l'historique des générations
        if 'generation_history' in st.session_state and st.session_state.generation_history:
            st.subheader("Historique")
            
            for i, gen in enumerate(reversed(st.session_state.generation_history[-5:])):  # Réduire à 5 dernières générations
                with st.expander(f"Génération #{len(st.session_state.generation_history) - i}"):
                    st.write(f"**État initial :** {gen['état_initial']}")
                    st.write(f"**Longueur :** {gen['longueur']} mots")
                    st.write(f"**Phrases :** {gen['phrases']}")
                    
                    # Afficher le texte généré avec des séparations claires entre les phrases
                    st.write("**Texte généré :**")
                    if "\n" in gen['texte']:
                        for j, sentence in enumerate(gen['texte'].split("\n")):
                            st.text(f"Phrase {j+1}: {sentence}")
                    else:
                        st.text(gen['texte'])
    else:
        st.info("Veuillez d'abord configurer le modèle dans l'onglet Configuration.")

# Section d'aide en bas de page, plus concise
with st.expander("À propos des chaînes de Markov"):
    st.markdown("""
    **Chaîne de Markov:** Processus stochastique où l'état futur ne dépend que de l'état présent.
    
    **Application pour la génération de texte**:
    - **États**: Chaque mot est un état
    - **Transitions**: Probabilités de passer d'un mot à un autre
    - **Matrice**: Représentation des probabilités de transition
    - **Génération**: Parcours aléatoire du graphe selon les probabilités
    """)