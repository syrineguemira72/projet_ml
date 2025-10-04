import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title="Projet ML - 3 Modèles",
    page_icon="🎯",
    layout="wide"
)

# Titre principal
st.title("🎯 Projet ML - Comparaison des 3 Modèles")
st.markdown("---")

# Chargement des modèles
@st.cache_resource
def load_models():
    try:
        model1 = joblib.load('models/modele_groupe1.pkl')
        model2 = joblib.load('models/modele_groupe2.pkl')
        model3 = joblib.load('models/modele_groupe3.pkl')
        return model1, model2, model3
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles: {e}")
        return None, None, None

# Sidebar pour les inputs
st.sidebar.header("📊 Paramètres d'Entrée")

# Exemple de features
feature1 = st.sidebar.slider("Feature 1", 0.0, 10.0, 5.0, 0.1)
feature2 = st.sidebar.slider("Feature 2", 0.0, 10.0, 5.0, 0.1)
feature3 = st.sidebar.slider("Feature 3", 0.0, 10.0, 5.0, 0.1)

# Bouton de prédiction
if st.sidebar.button("🎲 Lancer les Prédictions", type="primary"):
    model1, model2, model3 = load_models()
    
    if model1 and model2 and model3:
        input_data = np.array([[feature1, feature2, feature3]])
        
        with st.spinner('Calcul des prédictions...'):
            pred1 = model1.predict(input_data)
            pred2 = model2.predict(input_data)
            pred3 = model3.predict(input_data)
        
        st.success("Prédictions terminées !")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🧠 Modèle Groupe 1", f"{pred1[0]:.2f}")
        
        with col2:
            st.metric("🧠 Modèle Groupe 2", f"{pred2[0]:.2f}")
        
        with col3:
            st.metric("🧠 Modèle Groupe 3", f"{pred3[0]:.2f}")
        
        # Comparaison
        st.markdown("### 📈 Comparaison des Prédictions")
        comparison_data = pd.DataFrame({
            'Modèle': ['Groupe 1', 'Groupe 2', 'Groupe 3'],
            'Prédiction': [pred1[0], pred2[0], pred3[0]]
        })
        st.bar_chart(comparison_data.set_index('Modèle'))
        
    else:
        st.error("❌ Impossible de charger les modèles")

with st.expander("ℹ️ Instructions"):
    st.markdown("""
    1. **Ajustez les paramètres** dans la sidebar
    2. **Cliquez sur le bouton** pour lancer les prédictions
    3. **Comparez les résultats** des 3 modèles
    """)