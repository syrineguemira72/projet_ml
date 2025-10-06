import streamlit as st
import joblib
import numpy as np
import pandas as pd
import pickle

# Configuration de la page
st.set_page_config(
    page_title="Système Multi-Modèles ML",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement de tous les modèles
@st.cache_resource
def load_all_models():
    try:
        # Modèle de productivité (existant)
        model_productivite = joblib.load('models/modele_productivite.pkl')
        preprocesseurs = joblib.load('models/preprocesseurs.pkl')
       
        # Chargement du fichier over_time.pkl
        with open('models/over_time.pkl', 'rb') as f:
            overtime_data = pickle.load(f)
        
        # Vérification du type de données
        st.sidebar.info(f"Type over_time: {type(overtime_data)}")
        
        return {
            'productivite': model_productivite,
            'preprocesseurs': preprocesseurs,
            'overtime_data': overtime_data
        }
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles: {e}")
        return None

# Sidebar principale pour la navigation
st.sidebar.title("🧠 Navigation des Modèles")
st.sidebar.markdown("---")

# Sélection du modèle
model_choice = st.sidebar.radio(
    "Choisissez le modèle à utiliser:",
    ["🏭 Modèle Productivité", "⏱️ Analyse Heures Supplémentaires", "🔬 Modèle Groupe 2", "🧪 Modèle Groupe 3"],
    index=0
)

# Charger tous les modèles une fois
models_dict = load_all_models()

if models_dict is None:
    st.error("❌ Impossible de charger les modèles. Vérifiez les fichiers dans le dossier 'models/'.")
    st.stop()

# ============================================================================
# INTERFACE 1: MODÈLE PRODUCTIVITÉ
# ============================================================================
if model_choice == "🏭 Modèle Productivité":
    st.title("🏭 Modèle de Prédiction de Productivité")
    st.markdown("---")
   
    st.sidebar.header("📊 Paramètres de Production")
   
    # Features pour le modèle de productivité
    col1, col2 = st.columns(2)
   
    with col1:
        team = st.selectbox("Équipe (Team)", options=list(range(1, 13)), key="prod_team")
        targeted_productivity = st.slider("Productivité Cible", 0.0, 1.0, 0.8, 0.01, key="prod_target")
        smv = st.slider("SMV (Standard Minute Value)", 0.0, 50.0, 25.0, 0.1, key="prod_smv")
        idle_men = st.slider("Hommes Inactifs", 0.0, 50.0, 5.0, 1.0, key="prod_idle")
        no_of_style_change = st.slider("Nombre de Changements de Style", 0, 20, 2, 1, key="prod_style")
   
    with col2:
        efficiency_ratio = st.slider("Ratio d'Efficacité", 0.0, 2.0, 1.0, 0.01, key="prod_eff")
        work_intensity = st.slider("Intensité de Travail", 0.0, 10.0, 5.0, 0.1, key="prod_intensity")
        smv_winsorized = st.slider("SMV Winsorisé", 0.0, 50.0, 25.0, 0.1, key="prod_smv_win")
        wip_winsorized = st.slider("WIP Winsorisé", 0.0, 10000.0, 5000.0, 100.0, key="prod_wip")
        incentive_winsorized = st.slider("Incitation Winsorisée", 0.0, 500.0, 100.0, 10.0, key="prod_inc")
        team_size_medium = st.selectbox("Taille d'Équipe Moyenne", options=[0, 1], key="prod_team_size")
   
    # Bouton de prédiction
    if st.button("🎲 Prédire la Productivité", type="primary", key="prod_btn"):
        input_data = np.array([[
            team, targeted_productivity, smv, idle_men, no_of_style_change,
            efficiency_ratio, work_intensity, smv_winsorized, wip_winsorized,
            incentive_winsorized, team_size_medium
        ]])
       
        feature_names = [
            'team', 'targeted_productivity', 'smv', 'idle_men',
            'no_of_style_change', 'efficiency_ratio', 'work_intensity',
            'smv_winsorized', 'wip_winsorized', 'incentive_winsorized',
            'team_size_medium'
        ]
       
        input_df = pd.DataFrame(input_data, columns=feature_names)
       
        with st.spinner('Calcul de la prédiction...'):
            try:
                prediction = models_dict['productivite'].predict(input_df)
               
                st.success("✅ Prédiction terminée !")
               
                # Affichage des résultats
                result_col1, result_col2 = st.columns([1, 2])
               
                with result_col1:
                    st.metric(
                        "Productivité Prédite",
                        f"{prediction[0]:.3f}",
                        delta=f"{(prediction[0] - targeted_productivity):.3f} vs cible"
                    )
               
                with result_col2:
                    progress_value = max(0.0, min(1.0, prediction[0]))
                    st.progress(progress_value)
                    st.caption(f"Niveau de productivité: {progress_value*100:.1f}%")

            except Exception as e:
                st.error(f"Erreur lors de la prédiction: {e}")

# ============================================================================
# INTERFACE 2: ANALYSE HEURES SUPPLÉMENTAIRES (OVER_TIME)
# ============================================================================
elif model_choice == "⏱️ Analyse Heures Supplémentaires":
    st.title("⏱️ Analyse des Données Heures Supplémentaires")
    st.markdown("---")
    
    # Affichage des informations sur les données over_time
    overtime_data = models_dict['overtime_data']
    
    st.sidebar.header("📊 Informations sur les Données")
    
    # Affichage des métadonnées
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if hasattr(overtime_data, 'shape'):
            st.metric("Nombre d'observations", overtime_data.shape[0])
        else:
            st.metric("Type d'objet", type(overtime_data).__name__)
    
    with col2:
        if hasattr(overtime_data, 'shape'):
            st.metric("Nombre de variables", overtime_data.shape[1])
        else:
            st.metric("Est un DataFrame", isinstance(overtime_data, pd.DataFrame))
    
    with col3:
        if hasattr(overtime_data, 'columns'):
            st.metric("Colonnes disponibles", len(overtime_data.columns))
        else:
            st.metric("Données disponibles", "Oui")
    
    # Section d'exploration des données
    st.subheader("🔍 Exploration des Données")
    
    if isinstance(overtime_data, pd.DataFrame):
        # Affichage des premières lignes
        with st.expander("📋 Aperçu des données (5 premières lignes)"):
            st.dataframe(overtime_data.head())
        
        # Informations sur les colonnes
        with st.expander("📊 Informations sur les colonnes"):
            st.write("**Colonnes disponibles:**")
            for col in overtime_data.columns:
                st.write(f"- {col}: {overtime_data[col].dtype}")
                
            st.write("**Statistiques descriptives:**")
            st.dataframe(overtime_data.describe())
    
    # Interface d'analyse interactive
    st.subheader("📈 Analyse Interactive")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Variables de Production")
        smv = st.slider(
            "SMV (Standard Minute Value)", 
            min_value=0.0, 
            max_value=100.0, 
            value=25.0, 
            step=0.1,
            help="Temps standard alloué pour compléter une tâche"
        )
        
        no_of_workers = st.slider(
            "Nombre de Travailleurs", 
            min_value=1, 
            max_value=100, 
            value=50, 
            step=1,
            help="Effectif total des travailleurs"
        )
    
    with col2:
        st.subheader("🎯 Paramètres Additionnels")
        
        targeted_productivity = st.slider(
            "Productivité Cible", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.8, 
            step=0.01,
            help="Niveau de productivité visé"
        )
        
        work_intensity = st.slider(
            "Intensité de Travail", 
            min_value=0.0, 
            max_value=10.0, 
            value=5.0, 
            step=0.1,
            help="Niveau d'intensité du travail"
        )
    
    # Bouton d'analyse
    if st.button("📊 Analyser les Tendances", type="primary", key="analyze_btn"):
        try:
            with st.spinner('Analyse des tendances en cours...'):
                
                # Simulation d'analyse basée sur les données disponibles
                st.success("✅ Analyse terminée !")
                
                # Affichage des résultats simulés
                result_col1, result_col2, result_col3 = st.columns(3)
                
                with result_col1:
                    # Calcul basé sur SMV et nombre de travailleurs
                    heures_simulees = (smv * no_of_workers * work_intensity) / 60
                    st.metric(
                        "Heures Supplémentaires Estimées",
                        f"{heures_simulees:.1f} heures"
                    )
                
                with result_col2:
                    # Efficacité estimée
                    efficacite = targeted_productivity * 100
                    st.metric(
                        "Efficacité Estimée",
                        f"{efficacite:.1f}%"
                    )
                
                with result_col3:
                    # Coût estimé
                    cout_estime = heures_simulees * no_of_workers * 25  # 25€/heure
                    st.metric(
                        "Coût Estimé",
                        f"€{cout_estime:,.0f}"
                    )
                
                # Recommandations basées sur l'analyse
                st.subheader("💡 Recommandations")
                
                if heures_simulees > 20:
                    st.warning("""
                    **🔴 Attention - Niveau élevé d'heures supplémentaires détecté**
                    - Envisagez d'ajuster la charge de travail
                    - Évaluez l'embauche de personnel supplémentaire
                    - Revoyez les processus pour améliorer l'efficacité
                    """)
                elif heures_simulees > 10:
                    st.info("""
                    **🟡 Niveau moyen d'heures supplémentaires**
                    - Surveillez régulièrement la charge de travail
                    - Assurez un bon équilibre vie professionnelle
                    - Planifiez les pics d'activité à l'avance
                    """)
                else:
                    st.success("""
                    **🟢 Niveau faible d'heures supplémentaires**
                    - Bon équilibre maintenu
                    - Continuez les bonnes pratiques actuelles
                    """)
                
                # Visualisations supplémentaires si c'est un DataFrame
                if isinstance(overtime_data, pd.DataFrame):
                    st.subheader("📈 Visualisations des Données Réelles")
                    
                    # Sélection de colonnes pour visualisation
                    numeric_cols = overtime_data.select_dtypes(include=[np.number]).columns
                    
                    if len(numeric_cols) > 0:
                        col_viz1, col_viz2 = st.columns(2)
                        
                        with col_viz1:
                            selected_col = st.selectbox(
                                "Choisissez une colonne à visualiser:",
                                options=numeric_cols
                            )
                            
                            if selected_col:
                                fig, ax = plt.subplots()
                                overtime_data[selected_col].hist(ax=ax, bins=20)
                                ax.set_title(f'Distribution de {selected_col}')
                                st.pyplot(fig)
                        
                        with col_viz2:
                            st.write("**Résumé statistique:**")
                            if selected_col:
                                st.dataframe(overtime_data[selected_col].describe())
                    
        except Exception as e:
            st.error(f"Erreur lors de l'analyse: {e}")
    
    # Section d'export des données
    if isinstance(overtime_data, pd.DataFrame):
        st.subheader("💾 Export des Données")
        
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            if st.button("📥 Télécharger les données complètes"):
                csv = overtime_data.to_csv(index=False)
                st.download_button(
                    label="📋 Télécharger CSV",
                    data=csv,
                    file_name="donnees_heures_supplementaires.csv",
                    mime="text/csv"
                )
        
        with col_export2:
            if st.button("📊 Générer rapport statistique"):
                with st.expander("📈 Rapport Statistique Complet"):
                    st.write("**Statistiques descriptives complètes:**")
                    st.dataframe(overtime_data.describe(include='all'))

# ============================================================================
# PIED DE PAGE COMMUN
# ============================================================================
st.markdown("---")
st.markdown("### 📊 Tableau de Bord des Modèles")

# Aperçu des modèles chargés
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Modèle Productivité", "✅ Chargé" if models_dict.get('productivite') else "❌ Erreur")

with col2:
    overtime_status = "✅ Données" if models_dict.get('overtime_data') is not None else "❌ Erreur"
    st.metric("Données Heures Supp", overtime_status)

with col3:
    st.metric("Modèle Groupe 2", "🔜 Bientôt")

with col4:
    st.metric("Modèle Groupe 3", "🔜 Bientôt")

# Section d'aide
with st.expander("ℹ️ Guide d'Utilisation"):
    st.markdown("""
    ### Comment utiliser cette application :
   
    **🏭 Modèle Productivité** : Prédit l'efficacité de production industrielle
    - Utilise des variables comme SMV, WIP, productivité cible
    - Idéal pour l'optimisation manufacturière
   
    **⏱️ Analyse Heures Supplémentaires** : Explore les données sur les heures supplémentaires
    - Visualise les données disponibles
    - Analyse les tendances et patterns
    - Génère des recommandations basées sur l'analyse
   
    **🔬 Modèle Groupe 2** : [À venir]
   
    **🧪 Modèle Groupe 3** : [À venir]
    
    ### Variables Clés :
    - **SMV (Standard Minute Value)** : Temps standard pour compléter une tâche
    - **Over Time** : Données sur les heures supplémentaires
    - **No of Workers** : Effectif total des travailleurs
    """)

st.caption("Système Multi-Modèles ML • Développé avec Groupe 5")