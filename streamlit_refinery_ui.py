import streamlit as st

# Debe ser la primera llamada a Streamlit después de importar st
st.set_page_config(layout="wide", page_title="Simulador de Refinería")

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from empirical_data import EmpiricalDistributionManager
import logging # Import logging

# Configure logging if needed, or rely on refinery_calculations logging setup
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Importaciones de Módulos Locales ---
try:
    from refinery_calculations import (
        CrudeOil, DistillationCut,
        calculate_atmospheric_cuts, create_vacuum_feed_from_residue, calculate_vacuum_cuts,
        api_to_sg, sg_to_api, create_blend_from_crudes
    )
    from ui_logic import initialize_session_state, validate_crude_proportions, validate_cut_definitions_general
except ImportError as e:
    st.error(f"Error importando módulos: {e}. Asegúrese que todos los archivos .py estén en el mismo directorio.")
    st.stop()

# --- Funciones de UI y Manejo de Escenarios (Definidas ANTES de ser llamadas globalmente) ---

def generate_scenario_key(feed_name: str, feed_api: float, is_blend: bool, components_data: Optional[List[Dict[str, Any]]]=None) -> str:
    """Genera una clave para un escenario basado en la alimentación original."""
    if is_blend and components_data:
        sorted_components = sorted(components_data, key=lambda x: x.get('name', ''))
        key_string = "blend;" + ";".join([
            f"{c.get('name', '')}:{c.get('api', 0):.1f}:{c.get('proportion_vol', 0):.1f}"
            for c in sorted_components
        ])
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    else:
        # Ensure feed_name is a string before using replace
        safe_feed_name = str(feed_name) if feed_name is not None else "UnknownFeed"
        return f"{safe_feed_name}_{feed_api:.1f}"


def save_scenario_data(primary_key: str, scenario_type: str, scenario_name: str, data_to_save: Dict[str, Any], manager: EmpiricalDistributionManager):
    """Función auxiliar para guardar datos de escenario."""
    try:
        manager.save_scenario(
            primary_key=primary_key, scenario_type=scenario_type,
            scenario_name=scenario_name, distribution_data=data_to_save
        )
        st.success(f"✅ Escenario '{scenario_name}' (tipo: {scenario_type}) guardado exitosamente con clave principal: {primary_key}")
        # Don't rerun here, let the main flow continue after saving
        # st.rerun()
    except Exception as e:
        st.error(f"Error al guardar escenario '{scenario_name}': {e}")

def save_refinery_scenario_ui(original_feed: Optional[CrudeOil],
                              atmospheric_distillates: Optional[List[DistillationCut]], # Lista de objetos DistillationCut
                              vacuum_products: Optional[List[DistillationCut]], # Lista de objetos DistillationCut
                              atm_residue_as_vac_feed: Optional[DistillationCut], # Objeto DistillationCut
                              api_factor: Optional[float]): # Nombre del parámetro es api_factor
    """UI para guardar los resultados completos de la refinería como un escenario."""
    st.markdown("---")
    st.subheader("💾 Guardar Resultados Completos de Refinería como Escenario")
    # Check if original_feed exists and has a name and api attribute before proceeding
    if not original_feed or not hasattr(original_feed, 'name') or not hasattr(original_feed, 'api'):
         st.info("Datos de alimentación original incompletos. Calcule los resultados primero.")
         return
    # Further check if results exist (use the actual objects passed to the function)
    # Check if any list of products is not None and not empty, or if residue exists
    results_exist = (atmospheric_distillates and len(atmospheric_distillates) > 0) or \
                    (vacuum_products and len(vacuum_products) > 0) or \
                    (atm_residue_as_vac_feed is not None)

    if not results_exist:
        st.info("Calcule los resultados de la refinería para poder guardarlos.")
        return


    # Ensure original_feed.name is a string before using replace
    safe_feed_name_for_default = str(original_feed.name) if original_feed.name is not None else "UnknownFeed"
    default_scenario_name = f"Refineria_{safe_feed_name_for_default.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    # Generate a unique key for the text input based on feed properties
    text_input_key = f"save_refinery_scenario_name_{original_feed.name}_{original_feed.api:.1f}"

    scenario_name = st.text_input("Nombre del Escenario de Refinería", value=default_scenario_name, key=text_input_key)

    # Generate a unique key for the button based on feed properties
    button_key = f"save_refinery_scenario_button_{original_feed.name}_{original_feed.api:.1f}"

    if st.button("💾 Guardar Escenario de Refinería Completo", type="primary", use_container_width=True, key=button_key):
        scenario_name_to_save = scenario_name if scenario_name else default_scenario_name

        # Use the potentially edited objects from the session state for saving
        current_final_product_objects = st.session_state.get('all_final_cuts_objects_for_editing', [])
        # Also need potentially updated residue object if it was edited (though less likely)
        current_atm_residue_obj = st.session_state.get('calculated_atmospheric_residue', atm_residue_as_vac_feed)
        # Need the original calculated vacuum products list to determine origin
        current_vac_products_objs = st.session_state.get('calculated_vacuum_products', vacuum_products or [])


        final_products_list_to_save = []
        # Rebuild the list to save from the potentially edited objects in session state
        if current_final_product_objects:
             atm_res_yield_on_crude_frac = (current_atm_residue_obj.yield_vol_percent / 100.0) if current_atm_residue_obj and current_atm_residue_obj.yield_vol_percent is not None else 0.0
             vac_feed_for_check = st.session_state.get('vacuum_feed_object')

             for cut_obj in current_final_product_objects:
                 prod_dict = cut_obj.to_dict() # Get all properties from the object
                 # Determine origin based on whether it was part of the *original* calculated vacuum products list
                 is_vac_product = vac_feed_for_check and any(vp.name == cut_obj.name for vp in current_vac_products_objs)
                 prod_dict["Origen del Producto"] = "Vacío" if is_vac_product else "Atmosférico"

                 # Calculate yield on original crude
                 if prod_dict["Origen del Producto"] == "Vacío" and cut_obj.yield_vol_percent is not None:
                     prod_dict["Rend. Vol (%) en Crudo Orig."] = (cut_obj.yield_vol_percent / 100.0) * atm_res_yield_on_crude_frac * 100.0
                 else:
                     prod_dict["Rend. Vol (%) en Crudo Orig."] = cut_obj.yield_vol_percent # For atmospheric or if residue yield is zero

                 final_products_list_to_save.append(prod_dict)

        # Prepare feed data
        feed_data_to_save = {"name": original_feed.name, "api": original_feed.api, "sulfur": original_feed.sulfur_total_wt_percent,
                             "distillation_curve": list(zip(original_feed.distillation_volumes_percent, original_feed.distillation_temperatures_C)),
                             "is_blend": original_feed.is_blend}
        original_components = st.session_state.get("last_calculation_components_original_feed", [])

        # Prepare final payload
        data_payload: Dict[str, Any] = {
            "scenario_type": "refinery_run",
            "original_feed_properties": feed_data_to_save,
            "original_feed_components": original_components if original_feed.is_blend else [],
            "atmospheric_cut_definitions": st.session_state.atmospheric_cuts_definitions_df.to_dict('records') if not st.session_state.atmospheric_cuts_definitions_df.empty else [],
            "vacuum_cut_definitions": st.session_state.vacuum_cuts_definitions_df.to_dict('records') if not st.session_state.vacuum_cuts_definitions_df.empty else [],
            "final_products": final_products_list_to_save, # This list contains the potentially edited cuts
            "metadata": {"api_sensitivity_factor": api_factor, "description": f"Esc. refinería '{scenario_name_to_save}'."}
        }
        primary_key = generate_scenario_key(original_feed.name, original_feed.api, original_feed.is_blend, original_components)
        save_scenario_data(primary_key, "refinery_run", scenario_name_to_save, data_payload, st.session_state.empirical_mgr)
        st.rerun() # Rerun to refresh the scenario list


def show_scenario_management_ui(manager: EmpiricalDistributionManager, expander_title="Ver/Gestionar Todos los Escenarios Guardados"):
    """UI para ver y eliminar todos los escenarios guardados."""
    with st.expander(expander_title, expanded=True):
        st.markdown("Aquí puede ver y gestionar todos los escenarios empíricos guardados.")
        all_scenarios_dict = manager.list_all_scenarios()
        if not all_scenarios_dict: st.info("No hay escenarios guardados."); return
        scenario_data_for_df, delete_options = [], []
        for primary_key, data_for_pk in all_scenarios_dict.items():
            scenario_type = data_for_pk.get("scenario_type", "unknown")
            for scenario_name, scenario_content in data_for_pk.get("scenarios", {}).items():
                dist_data = scenario_content.get("distribution_data", {})
                feed_props = dist_data.get("original_feed_properties", {}) if scenario_type == "refinery_run" else dist_data
                display_name_col = feed_props.get("name", primary_key)
                display_api_col = feed_props.get("api", "N/A")
                if isinstance(display_api_col, (int, float)): display_api_col = f"{display_api_col:.1f}"
                try: last_updated_dt = datetime.fromisoformat(scenario_content.get("last_updated", "")); fmt_update = last_updated_dt.strftime("%Y-%m-%d %H:%M")
                except: fmt_update = scenario_content.get("last_updated", "N/A")
                scenario_data_for_df.append({"Tipo": scenario_type.replace("_", " ").capitalize(), "ID/Alim. Original": primary_key,
                                             "Nombre Alim.": display_name_col, "API Alim.": display_api_col,
                                             "Nombre Escenario": scenario_name, "Última Actualización": fmt_update})
                delete_options.append((primary_key, scenario_name, scenario_type))
        if scenario_data_for_df:
            st.dataframe(pd.DataFrame(scenario_data_for_df), hide_index=True, use_container_width=True,
                         column_config={"Tipo": st.column_config.TextColumn(width="small"), "ID/Alim. Original": st.column_config.TextColumn("ID / Alim. Orig.", width="medium"),
                                        "Nombre Alim.": st.column_config.TextColumn("Nombre Alim. Orig.", width="medium"), "API Alim.": st.column_config.TextColumn("API Alim. Orig.", width="small"),
                                        "Nombre Escenario": st.column_config.TextColumn(width="medium"), "Última Actualización": st.column_config.TextColumn(width="medium")})
        else: st.info("No hay escenarios para mostrar.")
        st.markdown("---"); st.subheader("🗑️ Eliminar un Escenario Específico")
        if delete_options:
            selected_to_delete = st.selectbox("Seleccione el escenario a eliminar:", options=delete_options,
                                              format_func=lambda x: f"Tipo: {x[2].replace('_',' ').capitalize()} | ID: {x[0]} | Esc: {x[1]}",
                                              key="delete_scenario_selector_all_v3")
            if st.button("Eliminar Escenario Seleccionado", type="primary", key="delete_scenario_button_all_v3"):
                if selected_to_delete:
                    pk_del, sn_del, st_del = selected_to_delete
                    if manager.delete_scenario(pk_del, sn_del): st.success(f"Escenario '{sn_del}' eliminado."); st.rerun()
                    else: st.error("No se pudo eliminar el escenario.")
        else: st.info("No hay escenarios para eliminar.")

# --- Inicialización de Estado ---
initialize_session_state()
if 'atmospheric_cuts_definitions_df' not in st.session_state:
    st.session_state.atmospheric_cuts_definitions_df = pd.DataFrame([
        {"Nombre del Corte": "Gases", "Temperatura Final (°C)": 20}, {"Nombre del Corte": "Nafta Liviana", "Temperatura Final (°C)": 90},
        {"Nombre del Corte": "Nafta Pesada", "Temperatura Final (°C)": 175}, {"Nombre del Corte": "Kerosene", "Temperatura Final (°C)": 235},
        {"Nombre del Corte": "Gasoil Liviano", "Temperatura Final (°C)": 290},{"Nombre del Corte": "Gasoil Pesado", "Temperatura Final (°C)": 350}])
if 'vacuum_cuts_definitions_df' not in st.session_state:
    st.session_state.vacuum_cuts_definitions_df = pd.DataFrame([
        {"Nombre del Corte": "Gasoil Liviano de Vacío (GOLV)", "Temperatura Final (°C)": 450},
        {"Nombre del Corte": "Gasoil Pesado de Vacío (GOPV)", "Temperatura Final (°C)": 550}])
if 'calculated_atmospheric_distillates' not in st.session_state: st.session_state.calculated_atmospheric_distillates = []
if 'calculated_atmospheric_residue' not in st.session_state: st.session_state.calculated_atmospheric_residue = None
if 'vacuum_feed_object' not in st.session_state: st.session_state.vacuum_feed_object = None
if 'calculated_vacuum_products' not in st.session_state: st.session_state.calculated_vacuum_products = []
if 'all_final_products_df_editable' not in st.session_state: st.session_state.all_final_products_df_editable = pd.DataFrame()
if 'all_final_cuts_objects_for_editing' not in st.session_state: st.session_state.all_final_cuts_objects_for_editing = []
if 'last_calculation_components_original_feed' not in st.session_state: st.session_state.last_calculation_components_original_feed = []
if 'crude_to_process' not in st.session_state: st.session_state.crude_to_process = None
if 'api_sensitivity_factor_display' not in st.session_state: st.session_state.api_sensitivity_factor_display = None
if 'api_sensitivity_factor' not in st.session_state: st.session_state.api_sensitivity_factor = 7.0

for comp in st.session_state.crude_components:
    comp_id = comp['id']
    if f"load_from_scenario_{comp_id}" not in st.session_state: st.session_state[f"load_from_scenario_{comp_id}"] = "Ingresar datos manualmente"
    if f"selected_scenario_key_{comp_id}" not in st.session_state: st.session_state[f"selected_scenario_key_{comp_id}"] = None
    if 'loaded_scenario_cuts' not in comp: comp['loaded_scenario_cuts'] = None
    if 'loaded_scenario_dist_curve' not in comp: comp['loaded_scenario_dist_curve'] = None
    if 'loaded_scenario_type' not in comp: comp['loaded_scenario_type'] = None

st.title("🛢️ Simulador de Refinería: Destilación Atmosférica y de Vacío")
tabs = st.tabs(["Alimentación y Escenarios", "Definición de Cortes", "Parámetros de Cálculo", "Resultados de Simulación"])

with tabs[0]: # Alimentación y Escenarios
    st.header("🛢️ Definición de Alimentación y Gestión de Escenarios")
    show_scenario_management_ui(st.session_state.empirical_mgr)
    st.markdown("---"); st.subheader("Componentes de la Alimentación Principal")
    num_crudes = len(st.session_state.crude_components)
    col1_btn_c, col2_btn_c = st.columns(2)
    with col1_btn_c:
        if st.button("➕ Añadir Componente a Alimentación", use_container_width=True, key="add_comp_button_main_v2"):
            new_id = st.session_state.next_crude_id
            st.session_state.crude_components.append({"id": new_id, "name": f"Componente {new_id}", "api": 30.0, "sulfur": 0.5,
                                                      "proportion_vol": 100.0 if num_crudes == 0 else 0.0,
                                                      "dist_curve": pd.DataFrame([{"Volumen (%)":v,"Temperatura (°C)":None} for v in [0,10,30,50,70,90,95]]),
                                                      "data_source_type":"manual", "loaded_scenario_cuts":None, "loaded_scenario_dist_curve":None, "loaded_scenario_type":None})
            st.session_state[f"load_from_scenario_{new_id}"] = "Ingresar datos manualmente"
            st.session_state[f"selected_scenario_key_{new_id}"] = None
            st.session_state.next_crude_id += 1; st.rerun()
    with col2_btn_c:
        if num_crudes > 0 and st.button("➖ Eliminar Último Componente", use_container_width=True, key="remove_comp_button_main_v2"):
            if st.session_state.crude_components: st.session_state.crude_components.pop(); st.rerun()

    crude_data_valid_overall = True
    component_crudes_for_processing = []
    all_scenarios_flat = st.session_state.empirical_mgr.list_all_scenarios_flat()

    for i, comp_state in enumerate(st.session_state.crude_components):
        comp_id = comp_state['id']
        st.markdown(f"--- \n### Componente {i+1}: {comp_state.get('name', f'Comp {comp_id}')}")
        data_source_key = f"load_from_scenario_{comp_id}"
        load_options = ["Ingresar datos manualmente"]
        scenario_map = {}
        for s_info in all_scenarios_flat:
            pk, sn, stype = s_info['primary_key'], s_info['scenario_name'], s_info['scenario_type']
            s_data = s_info.get('distribution_data', {})
            name_disp = s_data.get('original_feed_properties',s_data).get('name',pk) if stype == 'refinery_run' else s_data.get('name', pk)
            api_val = s_data.get('original_feed_properties',s_data).get('api','N/A') if stype == 'refinery_run' else s_data.get('api', 'N/A')
            api_disp = f"{api_val:.1f}" if isinstance(api_val,(float,int)) else str(api_val)
            label = f"Esc. {stype.replace('_',' ').capitalize()}: {name_disp} (API {api_disp}) - {sn}"
            load_options.append(label); scenario_map[label] = (pk, sn, stype)
        current_sel = st.session_state.get(data_source_key, "Ingresar datos manualmente")
        if current_sel != "Ingresar datos manualmente" and current_sel not in scenario_map:
            current_sel = "Ingresar datos manualmente"; st.session_state[data_source_key] = current_sel
        selected_source = st.selectbox(f"Fuente de datos Comp. {i+1}:", load_options, load_options.index(current_sel), key=data_source_key, help="Elija manual o un escenario guardado.")
        if selected_source != "Ingresar datos manualmente":
            if st.session_state.get(f"selected_scenario_key_{comp_id}") != selected_source:
                pk_load, sn_load, stype_load = scenario_map[selected_source]
                emp_data_payload = st.session_state.empirical_mgr.get_scenario_data(pk_load, sn_load)
                if emp_data_payload:
                    data_to_load = emp_data_payload.get('original_feed_properties', emp_data_payload) if stype_load == 'refinery_run' else emp_data_payload
                    comp_state["name"] = data_to_load.get("name", pk_load)
                    comp_state["api"] = data_to_load.get("api", 0.0)
                    comp_state["sulfur"] = data_to_load.get("sulfur", 0.0)
                    dist_curve = data_to_load.get("distillation_curve", [])
                    comp_state["dist_curve"] = pd.DataFrame([{"Volumen (%)":p[0],"Temperatura (°C)":p[1]} for p in dist_curve if isinstance(p,(list,tuple)) and len(p)==2]) if dist_curve else pd.DataFrame([{"Volumen (%)":v,"Temperatura (°C)":None} for v in [0,10,30,50,70,90,95]])
                    # When loading a 'refinery_run', the 'final_products' are the effective cuts to apply
                    comp_state["loaded_scenario_cuts"] = emp_data_payload.get("final_products") if stype_load == 'refinery_run' else emp_data_payload.get("cuts") if stype_load == 'crude' else None
                    comp_state["loaded_scenario_dist_curve"] = dist_curve
                    comp_state["loaded_scenario_type"] = stype_load
                    comp_state["data_source_type"] = "scenario"
                    st.session_state[f"selected_scenario_key_{comp_id}"] = selected_source
                    st.success(f"Datos cargados para Comp. {i+1} desde: {selected_source}"); st.rerun()
                else: st.error(f"No se pudo cargar: {selected_source}"); comp_state["data_source_type"]="manual"; comp_state["loaded_scenario_cuts"]=None; comp_state["loaded_scenario_dist_curve"]=None; comp_state["loaded_scenario_type"]=None
        else:
            comp_state["data_source_type"]="manual"; comp_state["loaded_scenario_cuts"]=None; comp_state["loaded_scenario_dist_curve"]=None; comp_state["loaded_scenario_type"]=None
            if st.session_state.get(f"selected_scenario_key_{comp_id}") is not None: st.session_state[f"selected_scenario_key_{comp_id}"]=None
        with st.container():
            if comp_state["data_source_type"] == "scenario": st.caption(f"Datos base cargados. Puede modificarlos.")
            c1,c2,c3,c4=st.columns(4)
            comp_state["name"]=c1.text_input("Nombre",comp_state["name"],key=f"n{comp_id}_main_v2")
            comp_state["api"]=c2.number_input("API",0.1,100.0,float(comp_state["api"]),0.1,"%.1f",key=f"a{comp_id}_main_v2")
            comp_state["sulfur"]=c3.number_input("Azufre %p",0.0,10.0,float(comp_state["sulfur"]),0.01,"%.2f",key=f"s{comp_id}_main_v2")
            comp_state["proportion_vol"]=c4.number_input("Prop. %vol",0.0,100.0,float(comp_state["proportion_vol"]),1.0,"%.1f",key=f"p{comp_id}_main_v2")
            st.markdown("##### Curva TBP")
            if not isinstance(comp_state.get("dist_curve"),pd.DataFrame): comp_state["dist_curve"]=pd.DataFrame([{"Volumen (%)":v,"Temperatura (°C)":None} for v in [0,10,30,50,70,90,95]])
            edited_df=st.data_editor(comp_state["dist_curve"],num_rows="dynamic",key=f"dc{comp_id}_main_v2",
                                     column_config={"Volumen (%)":st.column_config.NumberColumn(label="Volumen (%)",min_value=0.0,max_value=100.0,step=0.1,format="%.1f %%",required=True),
                                                    "Temperatura (°C)":st.column_config.NumberColumn(label="Temperatura (°C)",min_value=-50.0,max_value=1000.0,step=1.0,format="%d °C",required=True)},
                                     hide_index=True,use_container_width=True)
            df_clean=edited_df.dropna().copy()
            df_clean=df_clean[pd.to_numeric(df_clean["Volumen (%)"],errors='coerce').notnull() & pd.to_numeric(df_clean["Temperatura (°C)"],errors='coerce').notnull()]
            if not df_clean.empty:
                df_clean.loc[:,"Volumen (%)"]=pd.to_numeric(df_clean["Volumen (%)"]); df_clean.loc[:,"Temperatura (°C)"]=pd.to_numeric(df_clean["Temperatura (°C)"])
                df_clean=df_clean[(df_clean["Volumen (%)"]>=0)&(df_clean["Volumen (%)"]<=100)].sort_values("Volumen (%)").drop_duplicates("Volumen (%)",keep="last")
            comp_state["dist_curve"]=df_clean
            valid=True
            if len(df_clean)<2:st.warning(f"Comp. '{comp_state['name']}' necesita >=2 puntos.");valid=False
            if not df_clean.empty and 0.0 not in df_clean["Volumen (%)"].values:st.warning(f"Comp. '{comp_state['name']}' necesita IBP (0% vol).");valid=False
            if valid:
                dist_data_calc = list(zip(df_clean["Volumen (%)"].values, df_clean["Temperatura (°C)"].values))
                dist_data_key = comp_state.get('loaded_scenario_dist_curve') if comp_state.get('data_source_type')=='scenario' and comp_state.get('dist_curve').equals(pd.DataFrame([{"Volumen (%)":p[0],"Temperatura (°C)":p[1]} for p in comp_state.get('loaded_scenario_dist_curve',[]) if isinstance(p,(list,tuple)) and len(p)==2])) else dist_data_calc
                component_crudes_for_processing.append({'name':comp_state['name'],'api':comp_state['api'],'sulfur':comp_state['sulfur'],
                                                        'proportion_vol':comp_state['proportion_vol'],'distillation_data':dist_data_calc,
                                                        'distillation_data_for_key':dist_data_key,
                                                        'loaded_scenario_cuts':comp_state.get('loaded_scenario_cuts'), # Pass loaded cuts info
                                                        'loaded_scenario_type': comp_state.get('loaded_scenario_type')}) # Pass type
            else: crude_data_valid_overall=False
    proportions_ok = validate_crude_proportions()
    if crude_data_valid_overall and proportions_ok:
        if st.button("🚀 Procesar Alimentación y Calcular Cortes", type="primary", use_container_width=True, key="process_button_main_v2"):
            try:
                if not component_crudes_for_processing: st.error("No hay componentes válidos."); st.stop()
                st.session_state.last_calculation_components_original_feed = [{'name':c['name'],'api':c['api'], 'sulfur': c['sulfur'], 'proportion_vol':c['proportion_vol'], 'distillation_data':c['distillation_data_for_key']} for c in component_crudes_for_processing]
                current_feed_components_for_calc = [{'name':c['name'],'api':c['api'],'sulfur':c['sulfur'], 'proportion_vol':c['proportion_vol'],'distillation_data':c['distillation_data']} for c in component_crudes_for_processing]
                if len(current_feed_components_for_calc) > 1: st.session_state.crude_to_process = create_blend_from_crudes(current_feed_components_for_calc, verbose=True)
                else:
                    single_feed_data = current_feed_components_for_calc[0]
                    st.session_state.crude_to_process = CrudeOil(name=single_feed_data['name'], api_gravity=single_feed_data['api'], sulfur_content_wt_percent=single_feed_data['sulfur'], distillation_data_percent_vol_temp_C=single_feed_data['distillation_data'], verbose=True)

                # --- Check for empirical data to apply ---
                empirical_data_for_atm_tower = None
                # Only apply if exactly one component AND it was loaded from a scenario with cuts
                if len(component_crudes_for_processing) == 1:
                    single_comp_proc_data = component_crudes_for_processing[0]
                    if single_comp_proc_data.get('loaded_scenario_cuts') and single_comp_proc_data.get('data_source_type') == 'scenario':
                        cuts_to_apply = single_comp_proc_data['loaded_scenario_cuts']
                        # Ensure cuts_to_apply is a list of dicts (as expected by apply_empirical_distribution)
                        if isinstance(cuts_to_apply, list) and all(isinstance(item, dict) for item in cuts_to_apply):
                            empirical_data_for_atm_tower = {"distribution_data": {"cuts": cuts_to_apply }}
                            st.info(f"Utilizando distribución de productos empírica para '{single_comp_proc_data['name']}' en torre atmosférica (desde escenario '{single_comp_proc_data.get('loaded_scenario_type')}').")
                        else:
                            st.warning(f"Los datos de cortes cargados para '{single_comp_proc_data['name']}' no tienen el formato esperado (lista de diccionarios). Se calcularán teóricamente.")

                # --- Atmospheric Tower Calculation ---
                atm_defs = st.session_state.atmospheric_cuts_definitions_df
                if not validate_cut_definitions_general(atm_defs, "Cortes Atmosféricos"): st.stop()
                atm_cut_list = list(zip(atm_defs["Nombre del Corte"], pd.to_numeric(atm_defs["Temperatura Final (°C)"], errors='coerce')))

                api_sens_factor = st.session_state.get('api_sensitivity_factor', 7.0)

                st.session_state.calculated_atmospheric_distillates, st.session_state.calculated_atmospheric_residue = calculate_atmospheric_cuts(
                    crude_oil_feed=st.session_state.crude_to_process,
                    atmospheric_cut_definitions=atm_cut_list,
                    verbose=True,
                    api_sensitivity_factor=api_sens_factor,
                    empirical_data_for_crude=empirical_data_for_atm_tower
                )

                # --- Vacuum Tower Calculation ---
                st.session_state.vacuum_feed_object = None
                st.session_state.calculated_vacuum_products = []
                st.session_state.all_final_products_df_editable = pd.DataFrame()
                st.session_state.all_final_cuts_objects_for_editing = []

                if st.session_state.calculated_atmospheric_residue and st.session_state.calculated_atmospheric_residue.yield_vol_percent is not None and st.session_state.calculated_atmospheric_residue.yield_vol_percent > 1e-3:
                    st.session_state.vacuum_feed_object = create_vacuum_feed_from_residue(st.session_state.crude_to_process, st.session_state.calculated_atmospheric_residue, verbose=True)
                    if st.session_state.vacuum_feed_object:
                        vac_defs = st.session_state.vacuum_cuts_definitions_df
                        if not vac_defs.empty:
                            if not validate_cut_definitions_general(vac_defs, "Cortes de Vacío"): st.stop()
                            vac_cut_list = list(zip(vac_defs["Nombre del Corte"], pd.to_numeric(vac_defs["Temperatura Final (°C)"], errors='coerce')))
                            st.session_state.calculated_vacuum_products = calculate_vacuum_cuts(st.session_state.vacuum_feed_object, vac_cut_list, True, api_sens_factor)

                # --- Consolidate Final Products ---
                temp_all_final_cuts_objects = []
                if st.session_state.calculated_atmospheric_distillates:
                    temp_all_final_cuts_objects.extend(st.session_state.calculated_atmospheric_distillates)

                # Always add atmospheric residue if it exists and has a yield > 0
                if st.session_state.calculated_atmospheric_residue and \
                   st.session_state.calculated_atmospheric_residue.yield_vol_percent is not None and \
                   st.session_state.calculated_atmospheric_residue.yield_vol_percent > 1e-6: # Use a small tolerance
                    temp_all_final_cuts_objects.append(st.session_state.calculated_atmospheric_residue)

                if st.session_state.calculated_vacuum_products: # Then add vacuum products
                    temp_all_final_cuts_objects.extend(st.session_state.calculated_vacuum_products)
                # --- END MODIFIED LOGIC ---

                # Filter out any potential None objects
                st.session_state.all_final_cuts_objects_for_editing = [c for c in temp_all_final_cuts_objects if c]

                # --- Prepare DataFrame for Display/Editing ---
                final_products_data_for_df = []
                atm_res_yield_on_crude_frac = (st.session_state.calculated_atmospheric_residue.yield_vol_percent / 100.0) if st.session_state.calculated_atmospheric_residue and st.session_state.calculated_atmospheric_residue.yield_vol_percent is not None else 0.0
                for cut_obj in st.session_state.all_final_cuts_objects_for_editing:
                    prod_dict = cut_obj.to_dict()
                    # Determine if it's a vacuum product based on whether its name is in the list of calculated vacuum products
                    is_vac_product = st.session_state.vacuum_feed_object and any(vp.name == cut_obj.name for vp in st.session_state.calculated_vacuum_products)
                    prod_dict["Origen del Producto"] = "Vacío" if is_vac_product else "Atmosférico"
                    if prod_dict["Origen del Producto"] == "Vacío" and cut_obj.yield_vol_percent is not None:
                        prod_dict["Rend. Vol (%) en Crudo Orig."] = (cut_obj.yield_vol_percent / 100.0) * atm_res_yield_on_crude_frac * 100.0
                    else: # This 'else' branch covers atmospheric distillates AND the atmospheric residue itself
                        prod_dict["Rend. Vol (%) en Crudo Orig."] = cut_obj.yield_vol_percent
                    final_products_data_for_df.append(prod_dict)
                st.session_state.all_final_products_df_editable = pd.DataFrame(final_products_data_for_df)
                st.session_state.api_sensitivity_factor_display = api_sens_factor
                st.success("✅ ¡Cálculos de refinería completados!")
            except ValueError as ve: st.error(f"Error Validación en cálculo: {ve}")
            except Exception as e: st.error(f"❌ Error durante cálculo: {e}"); logging.exception("Error during calculation:"); st.stop() # Log exception details
    elif not proportions_ok: pass
    else: st.warning("Corrija errores en datos de componentes antes de calcular.")

with tabs[1]: # Definición de Cortes
    st.header("📋 Definición de Cortes de Destilación")
    st.subheader("Cortes Atmosféricos")
    st.markdown("Defina productos de torre atmosférica y Temp. Fin (°C). Último corte define residuo atm. Deben ser únicos y crecientes.")
    edited_atm_cuts_df = st.data_editor(st.session_state.atmospheric_cuts_definitions_df, num_rows="dynamic", key="atm_cuts_editor_tab_main_v2",
                                        column_config={"Nombre del Corte": st.column_config.TextColumn(required=True),
                                                       "Temperatura Final (°C)": st.column_config.NumberColumn(label="Temp. Fin (°C)", required=True, min_value=0, format="%d")},
                                        hide_index=True, use_container_width=True)
    st.session_state.atmospheric_cuts_definitions_df = edited_atm_cuts_df
    if not edited_atm_cuts_df.empty:
        validate_cut_definitions_general(edited_atm_cuts_df, "Cortes Atmosféricos")
    st.subheader("Cortes de Vacío")
    st.markdown("Defina productos de torre de vacío y Temp. Fin (°C, TBP eq. atm.). Último corte define residuo de vacío. Únicos y crecientes.")
    edited_vac_cuts_df = st.data_editor(st.session_state.vacuum_cuts_definitions_df, num_rows="dynamic", key="vac_cuts_editor_tab_main_v2",
                                        column_config={"Nombre del Corte": st.column_config.TextColumn(required=True),
                                                       "Temperatura Final (°C)": st.column_config.NumberColumn(label="Temp. Fin (°C Eq.)", required=True, min_value=200, format="%d")},
                                        hide_index=True, use_container_width=True)
    st.session_state.vacuum_cuts_definitions_df = edited_vac_cuts_df
    if not edited_vac_cuts_df.empty:
        validate_cut_definitions_general(edited_vac_cuts_df, "Cortes de Vacío")

with tabs[2]: # Parámetros
    st.header("⚙️ Parámetros de Cálculo"); st.subheader("Factor de Sensibilidad API")
    cp1,cp2=st.columns([2,1]); cp1.markdown("Ajusta API de cortes. Default: 7.0."); cp2.number_input("Factor Sensibilidad API",0.1,20.0,st.session_state.get('api_sensitivity_factor',7.0),0.1,key='api_sensitivity_factor_input_main_v2',help="Default: 7.0")
    # Update session state when input changes
    st.session_state.api_sensitivity_factor = st.session_state.api_sensitivity_factor_input_main_v2

    if st.session_state.get('crude_to_process'):
        st.subheader("Curvas TBP de Alimentación Procesada")
        try:
            fig=go.Figure(); cp_obj=st.session_state.crude_to_process
            fig.add_trace(go.Scatter(x=cp_obj.distillation_volumes_percent,y=cp_obj.distillation_temperatures_C,mode='lines+markers',name=f"Alim: {cp_obj.name}",line=dict(color='royalblue',width=3),marker=dict(size=8)))
            if cp_obj.is_blend and hasattr(st.session_state,'last_calculation_components_original_feed'):
                for comp_d in st.session_state.last_calculation_components_original_feed:
                    dist_d=comp_d.get("distillation_data",[]);
                    if dist_d: df_c=pd.DataFrame(dist_d,columns=["Volumen (%)","Temperatura (°C)"]); fig.add_trace(go.Scatter(x=df_c["Volumen (%)"],y=df_c["Temperatura (°C)"],mode='lines',name=f"{comp_d.get('name','C')} ({comp_d.get('proportion_vol',0):.1f}%)",line=dict(dash='dot',width=1.5)))
            fig.update_layout(title="Curvas TBP Alimentación",xaxis_title="V(%)Rec.",yaxis_title="T(°C)",xaxis_range=[0,100],yaxis_rangemode='tozero',height=500,legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),margin=dict(l=50,r=50,t=80,b=50))
            st.plotly_chart(fig,use_container_width=True)
        except Exception as e:st.error(f"Error TBP: {e}")
    else:st.info("Calcule en 'Alimentación' para ver TBP.")

with tabs[3]: # Resultados
    st.header("📊 Resultados de Simulación")
    original_feed = st.session_state.get('crude_to_process')
    all_final_product_objects_for_display = st.session_state.get('all_final_cuts_objects_for_editing', [])
    api_factor_val = st.session_state.get('api_sensitivity_factor_display')

    if not original_feed or not all_final_product_objects_for_display:
        st.info("⬅️ Configure y calcule en 'Alimentación y Escenarios' para ver resultados.")
    else:
        st.subheader(f"Resumen Alimentación Original: '{original_feed.name}'")
        c1,c2=st.columns(2);c1.metric("API",f"{original_feed.api:.1f}");c1.metric("IBP(°C)",f"{original_feed.ibp_C:.1f}");
        c2.metric("SG",f"{original_feed.sg:.4f}");c2.metric("Azufre(%p)",f"{original_feed.sulfur_total_wt_percent:.4f}");
        if api_factor_val is not None:st.caption(f"Factor Sens. API: {api_factor_val:.1f}")

        st.markdown("---");st.subheader("Productos Finales de Refinería (Editable)")

        # Obtener el DataFrame del estado de sesión
        df_editable_unified = st.session_state.get('all_final_products_df_editable', pd.DataFrame())

        # Re-create DataFrame if empty but objects exist (e.g., after first calculation)
        if df_editable_unified.empty and all_final_product_objects_for_display :
            final_products_data_for_df_init = []
            atm_res_for_calc = st.session_state.get('calculated_atmospheric_residue')
            atm_res_yield_on_crude_frac_init = (atm_res_for_calc.yield_vol_percent / 100.0) if atm_res_for_calc and atm_res_for_calc.yield_vol_percent is not None else 0.0
            vac_feed_for_check = st.session_state.get('vacuum_feed_object') # Para chequear si un producto es de vacío

            for cut_obj_init in all_final_product_objects_for_display:
                prod_dict_init = cut_obj_init.to_dict()
                is_vac_product_init = vac_feed_for_check and any(vp.name == cut_obj_init.name for vp in st.session_state.get('calculated_vacuum_products', []))
                prod_dict_init["Origen del Producto"] = "Vacío" if is_vac_product_init else "Atmosférico"
                if prod_dict_init["Origen del Producto"] == "Vacío" and cut_obj_init.yield_vol_percent is not None:
                    prod_dict_init["Rend. Vol (%) en Crudo Orig."] = (cut_obj_init.yield_vol_percent / 100.0) * atm_res_yield_on_crude_frac_init * 100.0
                else:
                    prod_dict_init["Rend. Vol (%) en Crudo Orig."] = cut_obj_init.yield_vol_percent
                final_products_data_for_df_init.append(prod_dict_init)
            st.session_state.all_final_products_df_editable = pd.DataFrame(final_products_data_for_df_init)
            df_editable_unified = st.session_state.all_final_products_df_editable

        if not df_editable_unified.empty:
            df_display = df_editable_unified.copy()
            columns_to_hide = ["VABP (°C)", "API Corte"]
            columns_to_drop_existing = [col for col in columns_to_hide if col in df_display.columns]
            if columns_to_drop_existing:
                df_display = df_display.drop(columns=columns_to_drop_existing)

            dynamic_column_config = {}
            disabled_columns = ["Corte", "T Inicial (°C)", "T Final (°C)",
                                "Rend. Peso (%)", "SG Corte", "Azufre (ppm)",
                                "Rend. Vol (%) en Crudo Orig.", "Origen del Producto"]

            for col in df_display.columns:
                col_config = None
                try:
                    numeric_col = pd.to_numeric(df_display[col], errors='coerce')
                    is_numeric = pd.api.types.is_numeric_dtype(numeric_col) and not numeric_col.isnull().all()
                except Exception:
                    is_numeric = False
                is_disabled = col in disabled_columns

                if is_numeric:
                    col_config = st.column_config.NumberColumn(label=col, format="%.2f", disabled=is_disabled)
                    if col == "Rend. Vol (%)":
                         col_config = st.column_config.NumberColumn(label="Rend Vol % (s/Alim. Dir.)", format="%.2f", min_value=0.0, max_value=100.0, disabled=False)
                    elif col == "Azufre (%peso)":
                         col_config = st.column_config.NumberColumn(label="S %peso", format="%.2f", min_value=0.0, max_value=10.0, disabled=False)
                else:
                    col_config = st.column_config.TextColumn(label=col, disabled=is_disabled)
                if col_config:
                    dynamic_column_config[col] = col_config

            edited_df_unified_local = st.data_editor(
                df_display,
                column_config=dynamic_column_config,
                key="all_products_editor_main_v3",
                num_rows="fixed",
                hide_index=True,
                use_container_width=True
            )

            if st.button("🔄 Aplicar y Actualizar Cambios en Productos Finales", key="apply_all_edits_button_main_v3"):
                try:
                    updated_final_cuts_list = []
                    atm_res_obj_for_calc = st.session_state.get('calculated_atmospheric_residue')
                    vac_feed_obj_for_calc = st.session_state.get('vacuum_feed_object')
                    original_objects_list = st.session_state.get('all_final_cuts_objects_for_editing', [])

                    for index, row_data in edited_df_unified_local.iterrows():
                        original_obj = next((cut for cut in original_objects_list if cut.name == row_data["Corte"]), None)
                        if original_obj:
                            original_obj.yield_vol_percent = float(row_data.get("Rend. Vol (%)", original_obj.yield_vol_percent))
                            original_obj.set_sulfur_properties(float(row_data.get("Azufre (%peso)", original_obj.sulfur_cut_wt_percent)))
                            feed_sg_for_wt_calc = original_feed.sg
                            if row_data.get("Origen del Producto") == "Vacío" and vac_feed_obj_for_calc:
                                feed_sg_for_wt_calc = vac_feed_obj_for_calc.sg
                            if original_obj.sg_cut and feed_sg_for_wt_calc and original_obj.yield_vol_percent is not None:
                                 original_obj.yield_wt_percent = original_obj.yield_vol_percent * (original_obj.sg_cut / feed_sg_for_wt_calc) * (0.85 if original_obj.is_gas_cut else 1.0)
                            else:
                                 original_obj.yield_wt_percent = 0.0
                            updated_final_cuts_list.append(original_obj)
                        else:
                            logging.warning(f"No se encontró el objeto original para la fila {index} con corte '{row_data.get('Corte')}'. Se omitirá la actualización para esta fila.")
                            if index < len(original_objects_list):
                                 updated_final_cuts_list.append(original_objects_list[index])

                    st.session_state.all_final_cuts_objects_for_editing = updated_final_cuts_list
                    rebuilt_df_data = []
                    atm_res_yield_on_crude_frac_recalc = (atm_res_obj_for_calc.yield_vol_percent / 100.0) if atm_res_obj_for_calc and atm_res_obj_for_calc.yield_vol_percent is not None else 0.0

                    for cut_obj_recalc in updated_final_cuts_list:
                        prod_dict_recalc = cut_obj_recalc.to_dict()
                        is_vac_recalc = vac_feed_obj_for_calc and any(vp.name == cut_obj_recalc.name for vp in st.session_state.get('calculated_vacuum_products',[]))
                        prod_dict_recalc["Origen del Producto"] = "Vacío" if is_vac_recalc else "Atmosférico"
                        if prod_dict_recalc["Origen del Producto"] == "Vacío" and cut_obj_recalc.yield_vol_percent is not None:
                            prod_dict_recalc["Rend. Vol (%) en Crudo Orig."] = (cut_obj_recalc.yield_vol_percent / 100.0) * atm_res_yield_on_crude_frac_recalc * 100.0
                        else:
                            prod_dict_recalc["Rend. Vol (%) en Crudo Orig."] = cut_obj_recalc.yield_vol_percent
                        rebuilt_df_data.append(prod_dict_recalc)

                    st.session_state.all_final_products_df_editable = pd.DataFrame(rebuilt_df_data)
                    st.success("Cambios en productos finales aplicados.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error aplicando cambios: {e}")
                    logging.exception("Error applying edits:")

        else:
            st.info("No hay productos finales para mostrar o editar.")

        df_for_download = st.session_state.all_final_products_df_editable.copy()
        if not df_for_download.empty:
            safe_feed_name_for_file = str(original_feed.name).replace(' ', '_') if original_feed.name else "UnknownFeed"
            csv_data = df_for_download.to_csv(index=False, float_format='%.4f').encode('utf-8')
            st.download_button(label="📥 Descargar Productos Finales (CSV)", data=csv_data,
                               file_name=f"productos_finales_{safe_feed_name_for_file}.csv",
                               mime='text/csv', use_container_width=True, key="download_final_products_button")

        st.markdown("---"); st.subheader("Visualización de Rendimientos y Azufre")
        df_plot_unified = st.session_state.all_final_products_df_editable.copy()

        GRAPH_HEIGHT = 450
        Y_AXIS_PADDING_FACTOR = 1.45

        atm_plot_df = df_plot_unified[df_plot_unified["Origen del Producto"] == "Atmosférico"].copy()
        vac_plot_df = df_plot_unified[df_plot_unified["Origen del Producto"] == "Vacío"].copy()

        col_atm1, col_atm2 = st.columns(2)
        with col_atm1:
            if not atm_plot_df.empty and "Rend. Vol (%) en Crudo Orig." in atm_plot_df.columns:
                y_col = "Rend. Vol (%) en Crudo Orig."
                atm_plot_df["Rend_Plot"] = pd.to_numeric(atm_plot_df[y_col], errors='coerce').fillna(0)
                max_y_val = atm_plot_df["Rend_Plot"].max() if not atm_plot_df["Rend_Plot"].empty else 0
                if max_y_val == 0: max_y_val = 1
                fig_atm_y = px.bar(atm_plot_df, x="Corte", y="Rend_Plot", title="Rend. Vol. Atmosférico (s/Crudo Orig.)", text="Rend_Plot")
                fig_atm_y.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                fig_atm_y.update_layout(height=GRAPH_HEIGHT, margin=dict(t=60, b=40, l=40, r=20), yaxis_range=[0, max_y_val * Y_AXIS_PADDING_FACTOR])
                st.plotly_chart(fig_atm_y, use_container_width=True)
            else: st.caption("Sin datos rend. atm.")
        with col_atm2:
            if not atm_plot_df.empty and "Azufre (ppm)" in atm_plot_df.columns:
                y_col_s = "Azufre (ppm)"
                atm_plot_df["S_Plot"] = pd.to_numeric(atm_plot_df[y_col_s], errors='coerce').fillna(0)
                max_y_val_s = atm_plot_df["S_Plot"].max() if not atm_plot_df["S_Plot"].empty else 0
                if max_y_val_s == 0: max_y_val_s = 10
                fig_atm_s = px.bar(atm_plot_df, x="Corte", y="S_Plot", title="Azufre en Prod. Atmosféricos (ppm)", text="S_Plot")
                fig_atm_s.update_traces(texttemplate='%{text:.0f}', textposition='outside')
                fig_atm_s.update_layout(height=GRAPH_HEIGHT, margin=dict(t=60, b=40, l=40, r=20), yaxis_range=[0, max_y_val_s * Y_AXIS_PADDING_FACTOR])
                st.plotly_chart(fig_atm_s, use_container_width=True)
            else: st.caption("Sin datos azufre atm.")

        col_vac1, col_vac2 = st.columns(2)
        with col_vac1:
            if not vac_plot_df.empty and "Rend. Vol (%)" in vac_plot_df.columns:
                y_col_vac = "Rend. Vol (%)"
                vac_plot_df["Rend_Plot_Vac"] = pd.to_numeric(vac_plot_df[y_col_vac], errors='coerce').fillna(0)
                max_y_val_vac = vac_plot_df["Rend_Plot_Vac"].max() if not vac_plot_df["Rend_Plot_Vac"].empty else 0
                if max_y_val_vac == 0: max_y_val_vac = 1
                fig_vac_y = px.bar(vac_plot_df, x="Corte", y="Rend_Plot_Vac", title="Rend. Vol. Vacío (s/Alim. Vacío)", text="Rend_Plot_Vac")
                fig_vac_y.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                fig_vac_y.update_layout(height=GRAPH_HEIGHT, margin=dict(t=60, b=40, l=40, r=20), yaxis_range=[0, max_y_val_vac * Y_AXIS_PADDING_FACTOR])
                st.plotly_chart(fig_vac_y, use_container_width=True)
            else: st.caption("Sin datos rend. vacío.")
        with col_vac2:
            if not vac_plot_df.empty and "Azufre (ppm)" in vac_plot_df.columns:
                y_col_vac_s = "Azufre (ppm)"
                vac_plot_df["S_Plot_Vac"] = pd.to_numeric(vac_plot_df[y_col_vac_s], errors='coerce').fillna(0)
                max_y_val_vac_s = vac_plot_df["S_Plot_Vac"].max() if not vac_plot_df["S_Plot_Vac"].empty else 0
                if max_y_val_vac_s == 0: max_y_val_vac_s = 10
                fig_vac_s = px.bar(vac_plot_df, x="Corte", y="S_Plot_Vac", title="Azufre en Prod. Vacío (ppm)", text="S_Plot_Vac")
                fig_vac_s.update_traces(texttemplate='%{text:.0f}', textposition='outside')
                fig_vac_s.update_layout(height=GRAPH_HEIGHT, margin=dict(t=60, b=40, l=40, r=20), yaxis_range=[0, max_y_val_vac_s * Y_AXIS_PADDING_FACTOR])
                st.plotly_chart(fig_vac_s, use_container_width=True)
            else: st.caption("Sin datos azufre vacío.")

        # --- CURVAS DE DESTILACIÓN DE PRODUCTOS ELIMINADAS ---

        save_refinery_scenario_ui(original_feed,
                                  st.session_state.get('calculated_atmospheric_distillates', []),
                                  st.session_state.get('calculated_vacuum_products', []),
                                  st.session_state.get('calculated_atmospheric_residue'),
                                  api_factor_val)
# --- Fin del Script ---
