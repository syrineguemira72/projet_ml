import streamlit as st
import joblib
import numpy as np
import pandas as pd
import math

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
        model_workers = joblib.load('models/randomforest_no_of_workers.pkl')
        
        # Nouveaux modèles
      #  model_groupe1 = joblib.load('models/modele_groupe1.pkl')
       # model_groupe2 = joblib.load('models/modele_groupe2.pkl')
       # model_groupe3 = joblib.load('models/modele_groupe3.pkl')

        return {
            'productivite': model_productivite,
            'preprocesseurs': preprocesseurs,
            'workers': model_workers
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
    ["🏭 Modèle Productivité", "🧪 Modèle Groupe 2", "⚗️ Modèle Nombre des Workers"],
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
# INTERFACE 3: MODÈLE NOMBRE DES WORKERS
# ============================================================================
if model_choice == "⚗️ Modèle Nombre des Workers":
    st.title("⚗️ Modèle de Prédiction du Nombre des Workers")
    st.markdown("---")

    st.sidebar.header("📊 Paramètres de Production")

    # 3 features du modèle RandomForest
    SMV_MIN, SMV_MAX = 0.0, 52.94  # ou Q1=3.94, Q3=23.54 selon ce que tu veux
    OVERTIME_MIN, OVERTIME_MAX = 0.0, 6900.0
    STYLE_VALUES = [0, 1, 2]

    # ---- Dans l’interface ----
    col1, col2 = st.columns(2)

    with col1:
        smv = st.slider(
            "SMV (Standard Minute Value)",
            min_value=SMV_MIN,
            max_value=SMV_MAX,
            value=15.26,  # valeur par défaut (médiane)
            step=0.1
        )

        over_time = st.slider(
            "Over Time",
            min_value=OVERTIME_MIN,
            max_value=OVERTIME_MAX,
            value=3960.0,  # valeur par défaut (médiane)
            step=10.0
        )

    with col2:
        no_of_style_change = st.selectbox(
            "Nombre de Changements de Style",
            STYLE_VALUES,
            index=0
        )

    # Bouton de prédiction
    if st.button("🎲 Prédire le Nombre des Workers", type="primary"):
        input_df = pd.DataFrame([{
            'smv': smv,
            'over_time': over_time,
            'no_of_style_change': no_of_style_change
        }])

        with st.spinner("Calcul de la prédiction..."):
            try:
                prediction = models_dict['workers'].predict(input_df)
                st.success("✅ Prédiction terminée !")

                st.metric(
                    "Nombre de Workers Prédit",
                    f"{math.ceil(float(prediction[0]))} ouvriers"
                )

            except Exception as e:
                st.error(f"Erreur lors de la prédiction: {e}")


# ============================================================================
# PIED DE PAGE COMMUN
# ============================================================================
st.markdown("---")
st.markdown("### 📊 Tableau de Bord des Modèles")

# Aperçu des modèles chargés
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Modèle Productivité", "✅ Chargé" if models_dict['productivite'] else "❌ Erreur")



# Section d'aide
with st.expander("ℹ️ Guide d'Utilisation"):
    st.markdown("""
    ### Comment utiliser cette application :
    
    **🏭 Modèle Productivité** : Prédit l'efficacité de production industrielle
    - Utilise des variables comme SMV, WIP, productivité cible
    - Ideal pour l'optimisation manufacturière
    
   
    
 
    """)

st.caption("Système Multi-Modèles ML • Développé avec Groupe 5")