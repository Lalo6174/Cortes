import streamlit as st
import pandas as pd
import numpy as np
from empirical_data import EmpiricalDistributionManager # Asegúrate que empirical_data.py esté presente

def initialize_session_state():
    # Inicializar el gestor de datos empíricos si no existe
    if 'empirical_mgr' not in st.session_state:
        st.session_state.empirical_mgr = EmpiricalDistributionManager()

    if 'crude_components' not in st.session_state:
        st.session_state.crude_components = [
            {
                "id": 1,
                "name": "Escalante (Ejemplo)", "api": 23.8, "sulfur": 0.22, "proportion_vol": 100.0,
                "dist_curve": pd.DataFrame([
                    {"Volumen (%)": 0, "Temperatura (°C)": 68}, {"Volumen (%)": 10, "Temperatura (°C)": 155},
                    {"Volumen (%)": 30, "Temperatura (°C)": 265}, {"Volumen (%)": 50, "Temperatura (°C)": 357},
                    {"Volumen (%)": 70, "Temperatura (°C)": 452}, {"Volumen (%)": 90, "Temperatura (°C)": 567},
                    {"Volumen (%)": 95, "Temperatura (°C)": 595}
                ]),
                "data_source_type": "manual", # Añadido para consistencia
                "loaded_scenario_cuts": None, # Añadido para consistencia
                "loaded_scenario_dist_curve": None, # Añadido para consistencia
                "loaded_scenario_type": None # Añadido para consistencia
            }
        ]
    if 'next_crude_id' not in st.session_state:
        st.session_state.next_crude_id = 2

    # Definiciones de cortes atmosféricos
    if 'atmospheric_cuts_definitions_df' not in st.session_state:
        st.session_state.atmospheric_cuts_definitions_df = pd.DataFrame([
            {"Nombre del Corte": "Gases", "Temperatura Final (°C)": 20},
            {"Nombre del Corte": "Nafta Liviana", "Temperatura Final (°C)": 90},
            {"Nombre del Corte": "Nafta Pesada", "Temperatura Final (°C)": 175},
            {"Nombre del Corte": "Kerosene", "Temperatura Final (°C)": 235},
            {"Nombre del Corte": "Gasóleo Atmosférico", "Temperatura Final (°C)": 350}
        ])
    # Definiciones de cortes de vacío
    if 'vacuum_cuts_definitions_df' not in st.session_state:
        st.session_state.vacuum_cuts_definitions_df = pd.DataFrame([
            {"Nombre del Corte": "Gasóleo Liviano de Vacío (LVGO)", "Temperatura Final (°C)": 450},
            {"Nombre del Corte": "Gasóleo Pesado de Vacío (HVGO)", "Temperatura Final (°C)": 550}
        ])

    if 'api_sensitivity_factor' not in st.session_state:
        st.session_state.api_sensitivity_factor = 7.0 # Default más común

def validate_crude_proportions():
    """Valida que la suma de las proporciones de los crudos sea 100%."""
    if not st.session_state.crude_components: # Si no hay componentes, no hay nada que validar aquí
        return True # O False si se requiere al menos un componente
    total_proportion = sum(float(c.get('proportion_vol', 0.0)) for c in st.session_state.crude_components)
    if not np.isclose(total_proportion, 100.0, rtol=1e-5, atol=1e-3): # Usar una tolerancia pequeña
        st.error(f"Error: La suma de las proporciones de los componentes ({total_proportion:.2f}%) debe ser 100%. Ajuste las proporciones.")
        # st.stop() # Evitar st.stop() aquí para permitir correcciones en la UI
        return False
    return True

def validate_cut_definitions_general(cuts_df: pd.DataFrame, df_name: str = "Definición de Cortes") -> bool:
    """
    Valida un DataFrame de definición de cortes.
    Asegura que no esté vacío, tenga las columnas necesarias, y las temperaturas sean válidas y crecientes.
    """
    if cuts_df.empty:
        st.error(f"Error: La tabla '{df_name}' está vacía. Por favor, defina al menos un corte.")
        return False

    required_columns = ["Nombre del Corte", "Temperatura Final (°C)"]
    if not all(col in cuts_df.columns for col in required_columns):
        st.error(f"Error: La tabla '{df_name}' debe tener las columnas: {', '.join(required_columns)}.")
        return False

    # Validar que los nombres de los cortes no estén vacíos
    if cuts_df["Nombre del Corte"].isnull().any() or (cuts_df["Nombre del Corte"] == "").any():
        st.error(f"Error: Todos los cortes en '{df_name}' deben tener un nombre.")
        return False

    # Validar temperaturas
    temps = pd.to_numeric(cuts_df["Temperatura Final (°C)"], errors='coerce')
    if temps.isnull().any():
        st.error(f"Error: Hay temperaturas no válidas o vacías en '{df_name}'.")
        return False

    # La validación de orden creciente y unicidad se hace en la UI para feedback inmediato,
    # pero podría reforzarse aquí si es necesario.

    return True


def export_results_to_csv(dataframe: pd.DataFrame, filename: str ="resultados_refineria.csv"):
    """Genera un botón para descargar un DataFrame como CSV."""
    if dataframe.empty:
        st.info("No hay datos para exportar.")
        return
    csv = dataframe.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar Resultados como CSV",
        data=csv,
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )