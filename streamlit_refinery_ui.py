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
import logging
import io # Necesario para leer el archivo cargado

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

# --- Inicialización de Estado ---
initialize_session_state() # Llama a la función de ui_logic.py

for comp in st.session_state.crude_components:
    comp_id = comp['id']
    if f"load_from_scenario_{comp_id}" not in st.session_state: st.session_state[f"load_from_scenario_{comp_id}"] = "Ingresar datos manualmente"
    if f"selected_scenario_key_{comp_id}" not in st.session_state: st.session_state[f"selected_scenario_key_{comp_id}"] = None
    if 'loaded_scenario_cuts' not in comp: comp['loaded_scenario_cuts'] = None
    if 'loaded_scenario_dist_curve' not in comp: comp['loaded_scenario_dist_curve'] = None
    if 'loaded_scenario_type' not in comp: comp['loaded_scenario_type'] = None
    if 'distillation_curve_type' not in comp: comp['distillation_curve_type'] = "TBP"


# --- Funciones de UI y Manejo de Escenarios ---

def generate_scenario_key(feed_name: str, feed_api: float, is_blend: bool, components_data: Optional[List[Dict[str, Any]]]=None) -> str:
    """Genera una clave para un escenario basado en la alimentación original."""
    if is_blend and components_data:
        sorted_components = sorted(components_data, key=lambda x: x.get('name', ''))
        key_string = "blend;" + ";".join([
            f"{c.get('name', '')}:{c.get('api', 0):.1f}:{c.get('proportion_vol', 0):.1f}:{c.get('distillation_curve_type','TBP')}"
            for c in sorted_components
        ])
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    else:
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
    except Exception as e:
        st.error(f"Error al guardar escenario '{scenario_name}': {e}")

def save_refinery_scenario_ui(original_feed: Optional[CrudeOil], 
                              atmospheric_distillates: Optional[List[DistillationCut]],
                              vacuum_products: Optional[List[DistillationCut]],
                              atm_residue_as_vac_feed: Optional[DistillationCut],
                              api_factor: Optional[float]):
    st.markdown("---")
    st.subheader("💾 Guardar Resultados Completos de Refinería como Escenario")
    if not original_feed or not hasattr(original_feed, 'api_gravity') or not hasattr(original_feed, 'name'): 
         st.info("Datos de alimentación original incompletos. Calcule los resultados primero.")
         return

    results_exist = (atmospheric_distillates and len(atmospheric_distillates) > 0) or \
                    (vacuum_products and len(vacuum_products) > 0) or \
                    (atm_residue_as_vac_feed is not None)
    if not results_exist:
        st.info("Calcule los resultados de la refinería para poder guardarlos.")
        return

    safe_feed_name_for_default = str(original_feed.name) if original_feed.name is not None else "UnknownFeed"
    default_scenario_name = f"Refineria_{safe_feed_name_for_default.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    text_input_key = f"save_refinery_scenario_name_{original_feed.name}_{original_feed.api_gravity:.1f}" 
    scenario_name = st.text_input("Nombre del Escenario de Refinería", value=default_scenario_name, key=text_input_key)
    button_key = f"save_refinery_scenario_button_{original_feed.name}_{original_feed.api_gravity:.1f}" 

    if st.button("💾 Guardar Escenario de Refinería Completo", type="primary", use_container_width=True, key=button_key):
        scenario_name_to_save = scenario_name if scenario_name else default_scenario_name

        current_final_product_objects = st.session_state.get('all_final_cuts_objects_for_editing', [])
        current_atm_residue_obj = st.session_state.get('calculated_atmospheric_residue', atm_residue_as_vac_feed)
        current_vac_products_objs = st.session_state.get('calculated_vacuum_products', vacuum_products or [])
        final_products_list_to_save = []

        if current_final_product_objects:
             atm_res_yield_on_crude_frac = (current_atm_residue_obj.yield_vol_percent / 100.0) if current_atm_residue_obj and current_atm_residue_obj.yield_vol_percent is not None else 0.0
             vac_feed_for_check = st.session_state.get('vacuum_feed_object')
             for cut_obj in current_final_product_objects:
                 prod_dict = cut_obj.to_dict()
                 is_vac_product = vac_feed_for_check and any(vp.name == cut_obj.name for vp in current_vac_products_objs if vp) # Check if vp is not None
                 prod_dict["Origen del Producto"] = "Vacío" if is_vac_product else "Atmosférico"
                 if prod_dict["Origen del Producto"] == "Vacío" and cut_obj.yield_vol_percent is not None:
                     prod_dict["Rend. Vol (%) en Crudo Orig."] = (cut_obj.yield_vol_percent / 100.0) * atm_res_yield_on_crude_frac * 100.0
                 else:
                     prod_dict["Rend. Vol (%) en Crudo Orig."] = cut_obj.yield_vol_percent
                 final_products_list_to_save.append(prod_dict)
        
        feed_data_to_save = {
            "name": original_feed.name,
            "api": original_feed.api_gravity, 
            "sulfur": original_feed.sulfur_total_wt_percent,
            "original_distillation_data": original_feed.original_raw_distillation_data, 
            "original_distillation_curve_type": original_feed.original_distillation_curve_type,
            "tbp_distillation_curve": list(zip(original_feed.distillation_volumes_percent, original_feed.distillation_temperatures_C)) if original_feed.distillation_volumes_percent is not None and original_feed.distillation_temperatures_C is not None else [],
            "is_blend": original_feed.is_blend
        }
        original_components_details = []
        if original_feed.is_blend:
            original_components_details = st.session_state.get("last_calculation_components_original_feed", [])

        data_payload: Dict[str, Any] = {
            "scenario_type": "refinery_run",
            "original_feed_properties": feed_data_to_save,
            "original_feed_components": original_components_details,
            "atmospheric_cut_definitions": st.session_state.atmospheric_cuts_definitions_df.to_dict('records') if not st.session_state.atmospheric_cuts_definitions_df.empty else [],
            "vacuum_cut_definitions": st.session_state.vacuum_cuts_definitions_df.to_dict('records') if not st.session_state.vacuum_cuts_definitions_df.empty else [],
            "final_products": final_products_list_to_save,
            "metadata": {"api_sensitivity_factor": api_factor, "description": f"Esc. refinería '{scenario_name_to_save}'."}
        }
        primary_key_feed_name = original_feed.name 
        primary_key_feed_api = original_feed.api_gravity 
        primary_key_is_blend = original_feed.is_blend
        primary_key_components = original_components_details if original_feed.is_blend else None
        primary_key = generate_scenario_key(primary_key_feed_name, primary_key_feed_api, primary_key_is_blend, primary_key_components)
        save_scenario_data(primary_key, "refinery_run", scenario_name_to_save, data_payload, st.session_state.empirical_mgr)
        st.rerun()


def show_scenario_management_ui(manager: EmpiricalDistributionManager, expander_title="Ver/Gestionar Todos los Escenarios Guardados"):
    with st.expander(expander_title, expanded=False):
        st.markdown("Aquí puede ver y gestionar todos los escenarios guardados.")
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
                original_curve_type_disp = feed_props.get("original_distillation_curve_type", "TBP")
                feed_display_info = f"{display_name_col} (API {display_api_col}, Curva: {original_curve_type_disp})"
                try: last_updated_dt = datetime.fromisoformat(scenario_content.get("last_updated", "")); fmt_update = last_updated_dt.strftime("%Y-%m-%d %H:%M")
                except: fmt_update = scenario_content.get("last_updated", "N/A")
                scenario_data_for_df.append({
                    "Tipo Escenario": scenario_type.replace("_", " ").capitalize(),
                    "ID Primario": primary_key,
                    "Info Alimentación Original": feed_display_info,
                    "Nombre Escenario Guardado": scenario_name,
                    "Última Actualización": fmt_update
                })
                delete_options.append((primary_key, scenario_name, scenario_type, feed_display_info, scenario_name))

        if scenario_data_for_df:
            df_display_scenarios = pd.DataFrame(scenario_data_for_df)
            st.dataframe(df_display_scenarios, hide_index=True, use_container_width=True,
                         column_config={
                             "Tipo Escenario": st.column_config.TextColumn(width="small"),
                             "ID Primario": st.column_config.TextColumn("ID Primario", width="medium"),
                             "Info Alimentación Original": st.column_config.TextColumn("Alimentación Original", width="large"),
                             "Nombre Escenario Guardado": st.column_config.TextColumn("Nombre Escenario", width="medium"),
                             "Última Actualización": st.column_config.TextColumn(width="medium")
                         })
        else: st.info("No hay escenarios para mostrar.")
        st.markdown("---"); st.subheader("🗑️ Eliminar un Escenario Específico")
        if delete_options:
            selected_to_delete = st.selectbox(
                "Seleccione el escenario a eliminar:",
                options=delete_options,
                format_func=lambda x: f"Tipo: {x[2].replace('_',' ').capitalize()} | Alim: {x[3]} | Esc: {x[4]}",
                key="delete_scenario_selector_all_v4" 
            )
            if st.button("Eliminar Escenario Seleccionado", type="primary", key="delete_scenario_button_all_v4"):
                if selected_to_delete:
                    pk_del, sn_del, _, _, _ = selected_to_delete
                    if manager.delete_scenario(pk_del, sn_del):
                        st.success(f"Escenario '{sn_del}' eliminado."); st.rerun()
                    else: st.error("No se pudo eliminar el escenario.")
        else: st.info("No hay escenarios para eliminar.")

# --- Título y Pestañas ---
st.title("🛢️ Simulador de Refinería: Destilación Atmosférica y de Vacío")
tabs = st.tabs(["Alimentación y Escenarios", "Definición de Cortes", "Parámetros de Cálculo", "Resultados de Simulación"])

with tabs[0]: # Alimentación y Escenarios
    st.header("🛢️ Definición de Alimentación y Gestión de Escenarios")
    show_scenario_management_ui(st.session_state.empirical_mgr)
    st.markdown("---"); st.subheader("Componentes de la Alimentación Principal")

    num_crudes = len(st.session_state.crude_components)
    
    if st.button("➕ Añadir Componente a Alimentación", use_container_width=True, key="add_comp_button_main_v4"):
        new_id = st.session_state.next_crude_id
        st.session_state.crude_components.append({
            "id": new_id, "name": f"Componente {new_id}", "api": 30.0, "sulfur": 0.5,
            "proportion_vol": 0.0, 
            "distillation_curve_type": "TBP", 
            "dist_curve": pd.DataFrame([{"Volumen (%)":v,"Temperatura (°C)":None} for v in [0,10,30,50,70,90,95,100]]),
            "data_source_type":"manual", "loaded_scenario_cuts":None,
            "loaded_scenario_dist_curve":None, "loaded_scenario_type":None
        })
        st.session_state[f"load_from_scenario_{new_id}"] = "Ingresar datos manualmente"
        st.session_state[f"selected_scenario_key_{new_id}"] = None
        st.session_state.next_crude_id += 1
        st.rerun()

    if st.session_state.crude_components:
        current_total_proportion = sum(float(c.get('proportion_vol', 0.0)) for c in st.session_state.crude_components)
        sum_color = "green" if np.isclose(current_total_proportion, 100.0) else "red"
        st.markdown(f"**Suma de Proporciones Actual:** <span style='color:{sum_color}; font-weight:bold;'>{current_total_proportion:.2f}%</span> (Objetivo: 100%)", unsafe_allow_html=True)

    crude_data_valid_overall = True
    component_crudes_for_processing = []
    all_scenarios_flat = st.session_state.empirical_mgr.list_all_scenarios_flat()
    distillation_curve_type_options = ["TBP", "ASTM D86", "ASTM D1160", "ASTM D2887", "ASTM D7169"]

    components_to_iterate = list(st.session_state.crude_components) 

    for i, comp_state in enumerate(components_to_iterate):
        comp_id = comp_state['id']
        st.markdown(f"--- \n### Componente: {comp_state.get('name', f'Comp {comp_id}')}")
        
        col_header_1, col_header_2 = st.columns([0.8, 0.2])
        with col_header_1:
            pass 
        with col_header_2:
            if st.button(f"🗑️ Eliminar Componente", key=f"delete_comp_btn_{comp_id}", use_container_width=True, type="secondary"):
                st.session_state.crude_components = [c for c in st.session_state.crude_components if c['id'] != comp_id]
                for key_prefix in ["load_from_scenario_", "selected_scenario_key_"]:
                    if f"{key_prefix}{comp_id}" in st.session_state:
                        del st.session_state[f"{key_prefix}{comp_id}"]
                st.rerun()

        data_source_key = f"load_from_scenario_{comp_id}"
        load_options = ["Ingresar datos manualmente"]
        scenario_map = {} 
        for s_info in all_scenarios_flat:
            pk, sn, stype = s_info['primary_key'], s_info['scenario_name'], s_info['scenario_type']
            s_data = s_info.get('distribution_data', {})
            feed_props_for_display = s_data.get('original_feed_properties', s_data) if stype == 'refinery_run' else s_data
            name_disp = feed_props_for_display.get('name', pk)
            api_val = feed_props_for_display.get('api','N/A') 
            api_disp = f"{api_val:.1f}" if isinstance(api_val,(float,int)) else str(api_val)
            curve_type_disp = feed_props_for_display.get('original_distillation_curve_type', 'TBP') 
            label = f"Esc. {stype.replace('_',' ').capitalize()}: {name_disp} (API {api_disp}, Curva: {curve_type_disp}) - {sn}"
            load_options.append(label); scenario_map[label] = (pk, sn, stype)

        current_sel_source = st.session_state.get(data_source_key, "Ingresar datos manualmente")
        if current_sel_source != "Ingresar datos manualmente" and current_sel_source not in scenario_map:
            current_sel_source = "Ingresar datos manualmente"; st.session_state[data_source_key] = current_sel_source
        
        selected_source = st.selectbox(f"Fuente de datos para '{comp_state.get('name')}':", load_options, index=load_options.index(current_sel_source), key=data_source_key, help="Elija manual o un escenario guardado.")

        if selected_source != "Ingresar datos manualmente":
            if st.session_state.get(f"selected_scenario_key_{comp_id}") != selected_source: 
                pk_load, sn_load, stype_load = scenario_map[selected_source]
                emp_data_payload = st.session_state.empirical_mgr.get_scenario_data(pk_load, sn_load)
                if emp_data_payload:
                    data_to_load_base = emp_data_payload.get('original_feed_properties', emp_data_payload) if stype_load == 'refinery_run' else emp_data_payload
                    comp_state["name"] = data_to_load_base.get("name", pk_load)
                    comp_state["api"] = data_to_load_base.get("api", 30.0) 
                    comp_state["sulfur"] = data_to_load_base.get("sulfur", 0.5)
                    dist_curve_original_loaded = data_to_load_base.get("original_distillation_data", [])
                    comp_state["distillation_curve_type"] = data_to_load_base.get("original_distillation_curve_type", "TBP")
                    comp_state["dist_curve"] = pd.DataFrame(
                        [{"Volumen (%)":p[0],"Temperatura (°C)":p[1]} for p in dist_curve_original_loaded if isinstance(p,(list,tuple)) and len(p)==2]
                    ) if dist_curve_original_loaded else pd.DataFrame([{"Volumen (%)":v,"Temperatura (°C)":None} for v in [0,10,30,50,70,90,95,100]])
                    comp_state["loaded_scenario_cuts"] = emp_data_payload.get("final_products") if stype_load == 'refinery_run' else emp_data_payload.get("cuts") if stype_load == 'crude' else None
                    comp_state["loaded_scenario_dist_curve"] = dist_curve_original_loaded 
                    comp_state["loaded_scenario_type"] = stype_load
                    comp_state["data_source_type"] = "scenario"
                    st.session_state[f"selected_scenario_key_{comp_id}"] = selected_source
                    st.success(f"Datos cargados para '{comp_state.get('name')}' desde: {selected_source}"); st.rerun()
                else:
                    st.error(f"No se pudo cargar el escenario: {selected_source}")
                    comp_state["data_source_type"]="manual"; comp_state["loaded_scenario_cuts"]=None
                    comp_state["loaded_scenario_dist_curve"]=None; comp_state["loaded_scenario_type"]=None
        else: 
            comp_state["data_source_type"]="manual"; comp_state["loaded_scenario_cuts"]=None
            comp_state["loaded_scenario_dist_curve"]=None; comp_state["loaded_scenario_type"]=None
            if st.session_state.get(f"selected_scenario_key_{comp_id}") is not None:
                 st.session_state[f"selected_scenario_key_{comp_id}"]=None 

        with st.container():
            if comp_state["data_source_type"] == "scenario": st.caption(f"Datos base cargados desde escenario. Puede modificarlos o cargar un CSV.")
            
            c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 2]) 
            comp_state["name"]=c1.text_input("Nombre",comp_state.get("name", f"Comp {comp_id}"),key=f"n{comp_id}_main_v4")
            comp_state["api"]=c2.number_input("API",0.1,100.0,float(comp_state.get("api",30.0)),0.1,"%.1f",key=f"a{comp_id}_main_v4")
            comp_state["sulfur"]=c3.number_input("Azufre %p",0.0,10.0,float(comp_state.get("sulfur",0.5)),0.01,"%.2f",key=f"s{comp_id}_main_v4")
            comp_state["proportion_vol"]=c4.number_input("Prop. %vol",0.0,100.0,float(comp_state.get("proportion_vol",0.0)),0.1,"%.1f",key=f"p{comp_id}_main_v4") 
            current_curve_type = comp_state.get("distillation_curve_type", "TBP")
            if current_curve_type not in distillation_curve_type_options: current_curve_type = "TBP"
            comp_state["distillation_curve_type"] = c5.selectbox(
                "Tipo de Curva", options=distillation_curve_type_options,
                index=distillation_curve_type_options.index(current_curve_type),
                key=f"curvetype{comp_id}_main_v4",
                help="Seleccione el tipo de curva de destilación que está ingresando o que espera del CSV."
            )

            st.markdown(f"##### Curva de Destilación ({comp_state['distillation_curve_type']}) para '{comp_state.get('name')}'")
            
            uploader_key_string = f"dist_curve_uploader_comp_{comp_id}"
            
            uploaded_file = st.file_uploader(
                "Cargar CSV para Curva", 
                type=["csv"],
                key=uploader_key_string, 
                help="CSV con columnas 'Volumen (%)' y 'Temperatura (°C)' (o equivalentes en inglés)."
            )

            if uploaded_file is not None:
                df_uploaded = None
                encodings_to_try = ['utf-8-sig', 'utf-8', 'latin1', 'iso-8859-1', 'windows-1252']
                success_reading = False
                for encoding in encodings_to_try:
                    try:
                        uploaded_file.seek(0) 
                        temp_content_peek = uploaded_file.read(1024).decode(encoding, errors='ignore')
                        uploaded_file.seek(0)
                        decimal_separator = '.'
                        if ',' in temp_content_peek and '.' not in temp_content_peek: 
                            if temp_content_peek.count(',') > temp_content_peek.count('.'): 
                                decimal_separator = ','
                        
                        df_uploaded = pd.read_csv(uploaded_file, encoding=encoding, skipinitialspace=True, decimal=decimal_separator)
                        logging.info(f"CSV '{uploaded_file.name}' leído con encoding '{encoding}' y decimal '{decimal_separator}'. Columnas detectadas: {df_uploaded.columns.tolist()}")
                        success_reading = True
                        break 
                    except UnicodeDecodeError:
                        logging.warning(f"Fallo al decodificar CSV '{uploaded_file.name}' con encoding '{encoding}'.")
                        df_uploaded = None
                    except Exception as e_read: 
                        logging.error(f"Error general leyendo CSV '{uploaded_file.name}' con encoding '{encoding}': {e_read}")
                        df_uploaded = None
                
                if success_reading and df_uploaded is not None:
                    df_uploaded.columns = df_uploaded.columns.str.strip().str.replace('\ufeff', '', regex=False)
                    logging.info(f"Columnas después de strip y reemplazo de BOM: {df_uploaded.columns.tolist()}")

                    column_name_map = {
                        "Volume (%)": "Volumen (%)", "Vol (%)": "Volumen (%)", "Volumen": "Volumen (%)",
                        "Temperature (°C)": "Temperatura (°C)", "Temp (°C)": "Temperatura (°C)",
                        "Temperature (C)": "Temperatura (°C)", "Temperatura": "Temperatura (°C)",
                        "Volumen (%)": "Volumen (%)", "Temperatura (°C)": "Temperatura (°C)"
                    }
                    df_uploaded.rename(columns=column_name_map, inplace=True)
                    logging.info(f"Columnas después del mapeo de nombres: {df_uploaded.columns.tolist()}")

                    if "Volumen (%)" in df_uploaded.columns and "Temperatura (°C)" in df_uploaded.columns:
                        df_uploaded["Volumen (%)"] = pd.to_numeric(df_uploaded["Volumen (%)"], errors='coerce')
                        df_uploaded["Temperatura (°C)"] = pd.to_numeric(df_uploaded["Temperatura (°C)"], errors='coerce')
                        
                        df_uploaded = df_uploaded.dropna(subset=["Volumen (%)", "Temperatura (°C)"])
                        df_uploaded = df_uploaded[(df_uploaded["Volumen (%)"] >= 0) & (df_uploaded["Volumen (%)"] <= 100)]
                        df_uploaded = df_uploaded.sort_values("Volumen (%)").drop_duplicates("Volumen (%)", keep="last")
                        
                        if len(df_uploaded) >= 2:
                            comp_state["dist_curve"] = df_uploaded[["Volumen (%)", "Temperatura (°C)"]].copy()
                            st.success(f"Curva cargada desde '{uploaded_file.name}' para '{comp_state.get('name')}'.")
                        else:
                            st.warning(f"CSV '{uploaded_file.name}' sin datos válidos (>= 2 puntos después de procesar).")
                    else:
                        st.error(f"CSV '{uploaded_file.name}' no contiene las columnas esperadas ('Volumen (%)' y 'Temperatura (°C)') incluso después de intentar mapear nombres comunes. Columnas encontradas: {list(df_uploaded.columns)}")
                elif not success_reading: 
                    st.error(f"No se pudo leer o decodificar el archivo CSV '{uploaded_file.name}' con las codificaciones probadas. Verifique el formato, codificación y separador decimal del archivo.")


            if not isinstance(comp_state.get("dist_curve"),pd.DataFrame):
                comp_state["dist_curve"]=pd.DataFrame([{"Volumen (%)":v,"Temperatura (°C)":None} for v in [0,10,30,50,70,90,95,100]])

            edited_df_dist_curve=st.data_editor(comp_state["dist_curve"],num_rows="dynamic",key=f"dc_editor_comp_{comp_id}", 
                                     column_config={"Volumen (%)":st.column_config.NumberColumn(label="Volumen (%)",min_value=0.0,max_value=100.0,step=0.1,format="%.1f %%",required=True),
                                                    "Temperatura (°C)":st.column_config.NumberColumn(label="Temperatura (°C)",min_value=-100.0,max_value=1000.0,step=1.0,format="%d °C",required=True)},
                                     hide_index=True,use_container_width=True)
            
            df_processed_from_editor = edited_df_dist_curve.copy()
            if "Volumen (%)" in df_processed_from_editor.columns:
                df_processed_from_editor["Volumen (%)"] = pd.to_numeric(df_processed_from_editor["Volumen (%)"], errors='coerce')
            else: 
                df_processed_from_editor["Volumen (%)"] = pd.Series(dtype=float)
            if "Temperatura (°C)" in df_processed_from_editor.columns:
                df_processed_from_editor["Temperatura (°C)"] = pd.to_numeric(df_processed_from_editor["Temperatura (°C)"], errors='coerce')
            else: 
                df_processed_from_editor["Temperatura (°C)"] = pd.Series(dtype=float)

            df_clean_dist_manual = df_processed_from_editor.dropna(subset=["Volumen (%)", "Temperatura (°C)"])
            if not df_clean_dist_manual.empty:
                try:
                    df_clean_dist_manual = df_clean_dist_manual.astype({"Volumen (%)": float, "Temperatura (°C)": float})
                except Exception as e:
                    st.warning(f"No se pudo convertir columnas a float después del editor para {comp_state['name']}: {e}")
                    df_clean_dist_manual = pd.DataFrame(columns=["Volumen (%)", "Temperatura (°C)"]) 
                df_clean_dist_manual = df_clean_dist_manual[
                    (df_clean_dist_manual["Volumen (%)"] >= 0) & (df_clean_dist_manual["Volumen (%)"] <= 100)
                ].sort_values("Volumen (%)").drop_duplicates("Volumen (%)", keep="last")
            comp_state["dist_curve"] = df_clean_dist_manual 

            valid_dist_curve=True
            final_dist_curve_for_calc = comp_state["dist_curve"] 
            if not isinstance(final_dist_curve_for_calc, pd.DataFrame) or final_dist_curve_for_calc.empty:
                st.warning(f"Comp. '{comp_state['name']}': la curva de destilación está vacía o no es válida.")
                valid_dist_curve=False
            elif len(final_dist_curve_for_calc)<2:
                st.warning(f"Comp. '{comp_state['name']}': la curva de destilación necesita al menos 2 puntos válidos.");valid_dist_curve=False
            
            if valid_dist_curve and not final_dist_curve_for_calc.empty: 
                try:
                    vol_values = pd.to_numeric(final_dist_curve_for_calc["Volumen (%)"], errors='raise').values
                    temp_values = pd.to_numeric(final_dist_curve_for_calc["Temperatura (°C)"], errors='raise').values
                except ValueError:
                    st.error(f"Error crítico: las columnas de la curva de destilación para '{comp_state['name']}' no son numéricas incluso después del procesamiento.")
                    valid_dist_curve = False 
                    crude_data_valid_overall = False 
                
                if valid_dist_curve: 
                    if 0.0 not in vol_values: 
                        st.warning(f"Comp. '{comp_state['name']}': la curva debe incluir el punto IBP (0% vol).");valid_dist_curve=False
            
            if valid_dist_curve:
                dist_data_tuples_for_calc = list(zip(vol_values, temp_values))
                dist_data_for_key_and_save = dist_data_tuples_for_calc 
                component_crudes_for_processing.append({
                    'name':comp_state['name'], 'api':comp_state['api'], 'sulfur':comp_state['sulfur'],
                    'proportion_vol':comp_state['proportion_vol'],
                    'distillation_curve_type': comp_state['distillation_curve_type'], 
                    'distillation_data': dist_data_tuples_for_calc, 
                    'distillation_data_for_key': dist_data_for_key_and_save, 
                    'loaded_scenario_cuts':comp_state.get('loaded_scenario_cuts'),
                    'loaded_scenario_type': comp_state.get('loaded_scenario_type')
                })
            else: crude_data_valid_overall=False

    proportions_ok = validate_crude_proportions() 

    if crude_data_valid_overall and proportions_ok:
        if st.button("🚀 Procesar Alimentación y Calcular Cortes", type="primary", use_container_width=True, key="process_button_main_v4"):
            try:
                if not component_crudes_for_processing: st.error("No hay componentes válidos para procesar."); st.stop()
                st.session_state.last_calculation_components_original_feed = [
                    {'name':c['name'], 'api':c['api'], 'sulfur': c['sulfur'],
                     'proportion_vol':c['proportion_vol'],
                     'distillation_curve_type': c['distillation_curve_type'], 
                     'distillation_data':c['distillation_data_for_key']} 
                    for c in component_crudes_for_processing
                ]
                current_feed_components_for_calc = [
                    {'name':c['name'],'api':c['api'],'sulfur':c['sulfur'],
                     'proportion_vol':c['proportion_vol'],
                     'distillation_curve_type': c['distillation_curve_type'], 
                     'distillation_data':c['distillation_data']} 
                    for c in component_crudes_for_processing
                ]

                if len(current_feed_components_for_calc) > 1:
                    st.session_state.crude_to_process = create_blend_from_crudes(current_feed_components_for_calc, verbose=True)
                elif len(current_feed_components_for_calc) == 1: 
                    single_feed_data = current_feed_components_for_calc[0]
                    st.session_state.crude_to_process = CrudeOil(
                        name=single_feed_data['name'],
                        api_gravity=single_feed_data['api'],
                        sulfur_content_wt_percent=single_feed_data['sulfur'],
                        distillation_data_percent_vol_temp_C=single_feed_data['distillation_data'],
                        distillation_curve_type=single_feed_data['distillation_curve_type'], 
                        verbose=True
                    )
                else: 
                    st.error("No hay componentes definidos para procesar.")
                    st.stop()

                empirical_data_for_atm_tower = None
                if len(component_crudes_for_processing) == 1: 
                    single_comp_proc_data = component_crudes_for_processing[0]
                    if single_comp_proc_data.get('loaded_scenario_cuts') and single_comp_proc_data.get('data_source_type') == 'scenario':
                        cuts_to_apply = single_comp_proc_data['loaded_scenario_cuts']
                        if isinstance(cuts_to_apply, list) and all(isinstance(item, dict) for item in cuts_to_apply):
                            empirical_data_for_atm_tower = {"distribution_data": {"cuts": cuts_to_apply }}
                            st.info(f"Utilizando distribución de productos empírica para '{single_comp_proc_data['name']}' en torre atmosférica (desde escenario '{single_comp_proc_data.get('loaded_scenario_type')}').")
                        else:
                            st.warning(f"Los datos de cortes cargados para '{single_comp_proc_data['name']}' no tienen el formato esperado. Se calcularán teóricamente.")

                atm_defs = st.session_state.atmospheric_cuts_definitions_df
                if not validate_cut_definitions_general(atm_defs, "Cortes Atmosféricos"): st.stop()
                atm_cut_list = []
                if not atm_defs.empty: 
                    atm_cut_list = list(zip(atm_defs["Nombre del Corte"], pd.to_numeric(atm_defs["Temperatura Final (°C)"], errors='coerce')))

                api_sens_factor = st.session_state.get('api_sensitivity_factor', 7.0)
                st.session_state.calculated_atmospheric_distillates, st.session_state.calculated_atmospheric_residue = calculate_atmospheric_cuts(
                    crude_oil_feed=st.session_state.crude_to_process, 
                    atmospheric_cut_definitions=atm_cut_list, verbose=True,
                    api_sensitivity_factor=api_sens_factor,
                    empirical_data_for_crude=empirical_data_for_atm_tower
                )
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

                temp_all_final_cuts_objects = []
                if st.session_state.calculated_atmospheric_distillates:
                    temp_all_final_cuts_objects.extend(st.session_state.calculated_atmospheric_distillates)
                if st.session_state.calculated_atmospheric_residue and \
                   st.session_state.calculated_atmospheric_residue.yield_vol_percent is not None and \
                   st.session_state.calculated_atmospheric_residue.yield_vol_percent > 1e-6: 
                    if not st.session_state.calculated_vacuum_products:
                        temp_all_final_cuts_objects.append(st.session_state.calculated_atmospheric_residue)
                
                if st.session_state.calculated_vacuum_products: 
                    temp_all_final_cuts_objects.extend(st.session_state.calculated_vacuum_products)
                
                st.session_state.all_final_cuts_objects_for_editing = [c for c in temp_all_final_cuts_objects if c and hasattr(c, 'yield_vol_percent') and c.yield_vol_percent is not None and c.yield_vol_percent > 1e-6]

                final_products_data_for_df = []
                atm_res_yield_on_crude_frac = (st.session_state.calculated_atmospheric_residue.yield_vol_percent / 100.0) if st.session_state.calculated_atmospheric_residue and st.session_state.calculated_atmospheric_residue.yield_vol_percent is not None else 0.0
                for cut_obj in st.session_state.all_final_cuts_objects_for_editing:
                    prod_dict = cut_obj.to_dict()
                    is_vac_product = st.session_state.vacuum_feed_object and any(vp.name == cut_obj.name for vp in st.session_state.calculated_vacuum_products if vp) 
                    prod_dict["Origen del Producto"] = "Vacío" if is_vac_product else "Atmosférico"
                    
                    if prod_dict["Origen del Producto"] == "Vacío" and cut_obj.yield_vol_percent is not None:
                        prod_dict["Rend. Vol (%) en Crudo Orig."] = (cut_obj.yield_vol_percent / 100.0) * atm_res_yield_on_crude_frac * 100.0
                    else: 
                        prod_dict["Rend. Vol (%) en Crudo Orig."] = cut_obj.yield_vol_percent
                    
                    final_products_data_for_df.append(prod_dict)

                st.session_state.all_final_products_df_editable = pd.DataFrame(final_products_data_for_df)
                st.session_state.api_sensitivity_factor_display = api_sens_factor
                st.success("✅ ¡Cálculos de refinería completados!")
                st.session_state.active_tab = "Resultados de Simulación" 
                st.rerun() 
            except ValueError as ve: st.error(f"Error Validación en cálculo: {ve}")
            except Exception as e: st.error(f"❌ Error durante cálculo: {e}"); logging.exception("Error during calculation:"); st.stop()
    elif not proportions_ok: pass 
    else: st.warning("Corrija errores en datos de componentes antes de calcular.")


with tabs[1]: # Definición de Cortes
    st.header("📋 Definición de Cortes de Destilación")
    st.subheader("Cortes Atmosféricos")
    st.markdown("Defina productos de torre atmosférica y Temp. Fin (°C). Deben ser únicos y con temperaturas crecientes.")
    edited_atm_cuts_df = st.data_editor(st.session_state.atmospheric_cuts_definitions_df, num_rows="dynamic", key="atm_cuts_editor_tab_main_v4",
                                        column_config={"Nombre del Corte": st.column_config.TextColumn(required=True),
                                                       "Temperatura Final (°C)": st.column_config.NumberColumn(label="Temp. Fin (°C)", required=True, min_value=-50, format="%d")},
                                        hide_index=True, use_container_width=True)
    st.session_state.atmospheric_cuts_definitions_df = edited_atm_cuts_df
    validate_cut_definitions_general(edited_atm_cuts_df, "Cortes Atmosféricos")
    st.subheader("Cortes de Vacío")
    st.markdown("Defina productos de torre de vacío y Temp. Fin (°C, TBP eq. atm.). Deben ser únicos y con temperaturas crecientes.")
    edited_vac_cuts_df = st.data_editor(st.session_state.vacuum_cuts_definitions_df, num_rows="dynamic", key="vac_cuts_editor_tab_main_v4",
                                        column_config={"Nombre del Corte": st.column_config.TextColumn(required=True),
                                                       "Temperatura Final (°C)": st.column_config.NumberColumn(label="Temp. Fin (°C Eq.)", required=True, min_value=200, format="%d")},
                                        hide_index=True, use_container_width=True)
    st.session_state.vacuum_cuts_definitions_df = edited_vac_cuts_df
    validate_cut_definitions_general(edited_vac_cuts_df, "Cortes de Vacío")


with tabs[2]: # Parámetros
    st.header("⚙️ Parámetros de Cálculo"); st.subheader("Factor de Sensibilidad API")
    cp1,cp2=st.columns([2,1]); cp1.markdown("Ajusta API de cortes. Default: 7.0.");
    st.session_state.api_sensitivity_factor = cp2.number_input(
        "Factor Sensibilidad API",0.1,20.0,
        st.session_state.get('api_sensitivity_factor',7.0),0.1,
        key='api_sensitivity_factor_input_main_v4', 
        help="Default: 7.0"
    )
    if st.session_state.get('crude_to_process'):
        st.subheader("Curvas TBP de Alimentación Procesada")
        try:
            fig=go.Figure(); cp_obj=st.session_state.crude_to_process 
            if cp_obj.distillation_volumes_percent is not None and cp_obj.distillation_temperatures_C is not None and len(cp_obj.distillation_volumes_percent) > 0 :
                fig.add_trace(go.Scatter(x=cp_obj.distillation_volumes_percent,y=cp_obj.distillation_temperatures_C,mode='lines+markers',name=f"Alim. Procesada (TBP): {cp_obj.name}",line=dict(color='royalblue',width=3),marker=dict(size=8)))
            
            if cp_obj.is_blend and hasattr(st.session_state,'last_calculation_components_original_feed'):
                temp_component_objects_for_plot = []
                for comp_data_orig in st.session_state.last_calculation_components_original_feed:
                    try:
                        comp_obj_plot = CrudeOil(
                            name=str(comp_data_orig.get('name','?')),
                            api_gravity=float(comp_data_orig.get('api',0.0)),
                            sulfur_content_wt_percent=float(comp_data_orig.get('sulfur',0.0)),
                            distillation_data_percent_vol_temp_C=comp_data_orig.get('distillation_data',[]), 
                            distillation_curve_type=str(comp_data_orig.get('distillation_curve_type',"TBP")), 
                            verbose=False 
                        )
                        if comp_obj_plot.distillation_volumes_percent is not None and len(comp_obj_plot.distillation_volumes_percent) > 0:
                             temp_component_objects_for_plot.append({'obj': comp_obj_plot, 'proportion_vol': comp_data_orig.get('proportion_vol',0)})
                    except Exception as e_comp_plot:
                        logging.warning(f"No se pudo recrear componente {comp_data_orig.get('name','?')} para graficar TBP: {e_comp_plot}")
                
                for comp_plot_info in temp_component_objects_for_plot:
                    comp_o = comp_plot_info['obj']
                    fig.add_trace(go.Scatter(
                        x=comp_o.distillation_volumes_percent, 
                        y=comp_o.distillation_temperatures_C,  
                        mode='lines',
                        name=f"Comp. TBP: {comp_o.name} ({comp_plot_info.get('proportion_vol',0):.1f}%) (Orig: {comp_o.original_distillation_curve_type})",
                        line=dict(dash='dot',width=1.5))
                    )
            fig.update_layout(title="Curvas TBP Alimentación (Procesada y Componentes)",xaxis_title="V(%)Rec.",yaxis_title="T(°C)",xaxis_range=[0,100],yaxis_rangemode='tozero',height=500,legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),margin=dict(l=50,r=50,t=80,b=50))
            st.plotly_chart(fig,use_container_width=True)
        except Exception as e:st.error(f"Error graficando TBP: {e}")
    else:st.info("Calcule en 'Alimentación y Escenarios' para ver TBP.")

with tabs[3]: # Resultados
    st.header("📊 Resultados de Simulación")
    original_feed_processed = st.session_state.get('crude_to_process') 
    all_final_product_objects_for_display = st.session_state.get('all_final_cuts_objects_for_editing', [])
    api_factor_val = st.session_state.get('api_sensitivity_factor_display') 

    if not original_feed_processed or not all_final_product_objects_for_display:
        st.info("⬅️ Configure y calcule en 'Alimentación y Escenarios' para ver resultados.")
    else:
        st.subheader(f"Resumen Alimentación Original Procesada: '{original_feed_processed.name}'")
        col_feed1, col_feed2, col_feed3 = st.columns(3)
        col_feed1.metric("API (Alim. Procesada)",f"{original_feed_processed.api_gravity:.1f}") 
        col_feed1.metric("SG (Alim. Procesada)",f"{original_feed_processed.sg:.4f}" if original_feed_processed.sg else "N/A")
        col_feed2.metric("Azufre (%p, Alim. Procesada)",f"{original_feed_processed.sulfur_total_wt_percent:.4f}")
        col_feed2.metric(f"Tipo Curva Original Ingresada", f"{original_feed_processed.original_distillation_curve_type}")
        ibp_disp_res = f"{original_feed_processed.ibp_C:.1f}" if original_feed_processed.ibp_C is not None else "N/A"
        fbp_disp_res = f"{original_feed_processed.fbp_C:.1f}" if original_feed_processed.fbp_C is not None else "N/A"
        col_feed3.metric("IBP (°C, TBP Procesada)", ibp_disp_res)
        col_feed3.metric("FBP (°C, TBP Procesada)", fbp_disp_res)

        if api_factor_val is not None:st.caption(f"Factor Sens. API usado en cálculo: {api_factor_val:.1f}")

        st.markdown("---");st.subheader("Productos Finales de Refinería (Editable)")
        df_editable_unified = st.session_state.get('all_final_products_df_editable', pd.DataFrame())

        if df_editable_unified.empty and all_final_product_objects_for_display :
            final_products_data_for_df_init = []
            atm_res_for_calc = st.session_state.get('calculated_atmospheric_residue')
            atm_res_yield_on_crude_frac_init = (atm_res_for_calc.yield_vol_percent / 100.0) if atm_res_for_calc and atm_res_for_calc.yield_vol_percent is not None else 0.0
            vac_feed_for_check = st.session_state.get('vacuum_feed_object')
            for cut_obj_init in all_final_product_objects_for_display:
                if cut_obj_init is None: continue 
                prod_dict_init = cut_obj_init.to_dict()
                is_vac_product_init = vac_feed_for_check and any(vp.name == cut_obj_init.name for vp in st.session_state.get('calculated_vacuum_products', []) if vp)
                prod_dict_init["Origen del Producto"] = "Vacío" if is_vac_product_init else "Atmosférico"
                if prod_dict_init["Origen del Producto"] == "Vacío" and cut_obj_init.yield_vol_percent is not None:
                    prod_dict_init["Rend. Vol (%) en Crudo Orig."] = (cut_obj_init.yield_vol_percent / 100.0) * atm_res_yield_on_crude_frac_init * 100.0
                else:
                    prod_dict_init["Rend. Vol (%) en Crudo Orig."] = cut_obj_init.yield_vol_percent
                final_products_data_for_df_init.append(prod_dict_init)
            st.session_state.all_final_products_df_editable = pd.DataFrame(final_products_data_for_df_init)
            df_editable_unified = st.session_state.all_final_products_df_editable

        if not df_editable_unified.empty:
            df_display_results = df_editable_unified.copy()
            columns_to_display_config = {
                "Corte": st.column_config.TextColumn(label="Producto Final", disabled=True),
                "Origen del Producto": st.column_config.TextColumn(disabled=True),
                "T Inicial (°C)": st.column_config.NumberColumn(format="%.1f", disabled=True),
                "T Final (°C)": st.column_config.NumberColumn(format="%.1f", disabled=True),
                "Rend. Vol (%)": st.column_config.NumberColumn("Rend. Vol % (s/Alim. Dir.)", format="%.2f", min_value=0.0, max_value=100.0, disabled=False, help="Rendimiento volumétrico sobre la alimentación directa a la torre (atmosférica o vacío)"),
                "Rend. Vol (%) en Crudo Orig.": st.column_config.NumberColumn("Rend. Vol % (s/Crudo Orig.)",format="%.2f", disabled=True, help="Rendimiento volumétrico sobre el crudo original total"),
                "Rend. Peso (%)": st.column_config.NumberColumn(format="%.2f", disabled=True),
                "API Corte": st.column_config.NumberColumn(format="%.1f", disabled=True), 
                "SG Corte": st.column_config.NumberColumn(format="%.4f", disabled=True),
                "Azufre (%peso)": st.column_config.NumberColumn("S %peso", format="%.4f", min_value=0.0, max_value=10.0, disabled=False),
                "Azufre (ppm)": st.column_config.NumberColumn(format="%.0f", disabled=True),
                "VABP (°C)": st.column_config.NumberColumn(format="%.1f", disabled=True),
            }
            cols_to_show_in_editor = [col for col in columns_to_display_config.keys() if col in df_display_results.columns]
            df_for_editor = df_display_results[cols_to_show_in_editor]
            
            edited_df_unified_local = st.data_editor(
                df_for_editor, column_config=columns_to_display_config,
                key="all_products_editor_main_v5", num_rows="fixed", 
                hide_index=True, use_container_width=True
            )

            if st.button("🔄 Aplicar y Actualizar Cambios en Productos Finales", key="apply_all_edits_button_main_v5"):
                try:
                    updated_final_cuts_list = []
                    atm_res_obj_for_calc = st.session_state.get('calculated_atmospheric_residue')
                    vac_feed_obj_for_calc = st.session_state.get('vacuum_feed_object')
                    original_objects_list = st.session_state.get('all_final_cuts_objects_for_editing', [])
                    
                    for index, row_data_edited in edited_df_unified_local.iterrows():
                        original_obj = next((cut for cut in original_objects_list if cut and cut.name == row_data_edited["Corte"]), None) 
                        if original_obj:
                            original_obj.yield_vol_percent = float(row_data_edited.get("Rend. Vol (%)", original_obj.yield_vol_percent if original_obj.yield_vol_percent is not None else 0.0))
                            
                            new_sulfur_wt = float(row_data_edited.get("Azufre (%peso)", original_obj.sulfur_cut_wt_percent if original_obj.sulfur_cut_wt_percent is not None else 0.0))
                            if hasattr(original_obj, '_calculate_properties'): # Asumiendo que set_sulfur_properties no existe, recalculamos
                                original_obj.sulfur_cut_wt_percent = new_sulfur_wt # Setear directamente
                                original_obj.sulfur_cut_ppm = new_sulfur_wt * 10000
                                # Recalcular otras propiedades dependientes si es necesario, o ajustar _calculate_properties
                            
                            feed_sg_for_wt_calc = original_feed_processed.sg 
                            if row_data_edited.get("Origen del Producto") == "Vacío" and vac_feed_obj_for_calc:
                                feed_sg_for_wt_calc = vac_feed_obj_for_calc.sg 
                            
                            if original_obj.sg_cut and feed_sg_for_wt_calc and original_obj.yield_vol_percent is not None:
                                 density_corr_factor = 0.85 if original_obj.is_gas_cut else 1.0
                                 original_obj.yield_wt_percent = original_obj.yield_vol_percent * (original_obj.sg_cut / feed_sg_for_wt_calc) * density_corr_factor
                            else: original_obj.yield_wt_percent = 0.0 
                            updated_final_cuts_list.append(original_obj)
                        else: logging.warning(f"No se encontró el objeto original para '{row_data_edited.get('Corte')}'. Se omitirá.")
                    
                    st.session_state.all_final_cuts_objects_for_editing = updated_final_cuts_list
                    rebuilt_df_data = []
                    atm_res_yield_on_crude_frac_recalc = (atm_res_obj_for_calc.yield_vol_percent / 100.0) if atm_res_obj_for_calc and atm_res_obj_for_calc.yield_vol_percent is not None else 0.0
                    
                    for cut_obj_recalc in updated_final_cuts_list:
                        if cut_obj_recalc is None: continue
                        prod_dict_recalc = cut_obj_recalc.to_dict()
                        is_vac_recalc = vac_feed_obj_for_calc and any(vp.name == cut_obj_recalc.name for vp in st.session_state.get('calculated_vacuum_products',[]) if vp)
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
            safe_feed_name_for_file = str(original_feed_processed.name).replace(' ', '_').replace('(','').replace(')','') if original_feed_processed.name else "UnknownFeed"
            csv_data = df_for_download.to_csv(index=False, float_format='%.4f').encode('utf-8')
            st.download_button(label="📥 Descargar Productos Finales (CSV)", data=csv_data,
                               file_name=f"productos_finales_{safe_feed_name_for_file}.csv",
                               mime='text/csv', use_container_width=True, key="download_final_products_button_v2")

        st.markdown("---"); st.subheader("Visualización de Rendimientos y Azufre")
        df_plot_unified = st.session_state.all_final_products_df_editable.copy()
        GRAPH_HEIGHT = 450; Y_AXIS_PADDING_FACTOR = 1.45 
        
        for col_name in ["Rend. Vol (%) en Crudo Orig.", "Azufre (ppm)", "Rend. Vol (%)"]:
            if col_name in df_plot_unified.columns:
                df_plot_unified[col_name] = pd.to_numeric(df_plot_unified[col_name], errors='coerce').fillna(0)
            else: 
                df_plot_unified[col_name] = 0


        atm_plot_df = df_plot_unified[df_plot_unified["Origen del Producto"] == "Atmosférico"].copy()
        vac_plot_df = df_plot_unified[df_plot_unified["Origen del Producto"] == "Vacío"].copy()

        col_atm1, col_atm2 = st.columns(2)
        with col_atm1:
            if not atm_plot_df.empty:
                y_col_atm_yield = "Rend. Vol (%) en Crudo Orig."
                max_y_val = atm_plot_df[y_col_atm_yield].max() if not atm_plot_df[y_col_atm_yield].empty else 0
                if max_y_val == 0: max_y_val = 1 
                
                fig_atm_y = px.bar(atm_plot_df, x="Corte", y=y_col_atm_yield, 
                                   title="Rend. Vol. Atmosférico (s/Crudo Orig.)", 
                                   text_auto='.2f', 
                                   labels={"Corte": "Producto Atmosférico", y_col_atm_yield: "Rendimiento Vol. (%)"}) 
                fig_atm_y.update_traces(textposition='outside', textfont_color='white') # Texto blanco y fuera
                fig_atm_y.update_layout(height=GRAPH_HEIGHT, margin=dict(t=60, b=40, l=40, r=20), 
                                        yaxis_title="Rendimiento Vol. (%) s/Crudo", 
                                        yaxis_range=[0, max_y_val * Y_AXIS_PADDING_FACTOR])
                st.plotly_chart(fig_atm_y, use_container_width=True)
            else: st.caption("Sin datos de rendimiento atmosférico para graficar.")
        
        with col_atm2:
            if not atm_plot_df.empty:
                y_col_atm_sulfur = "Azufre (ppm)"
                max_y_val_s = atm_plot_df[y_col_atm_sulfur].max() if not atm_plot_df[y_col_atm_sulfur].empty else 0
                if max_y_val_s == 0: max_y_val_s = 10 
                
                fig_atm_s = px.bar(atm_plot_df, x="Corte", y=y_col_atm_sulfur, 
                                   title="Azufre en Prod. Atmosféricos (ppm)", 
                                   text_auto='.0f',
                                   labels={"Corte": "Producto Atmosférico", y_col_atm_sulfur: "Azufre (ppm)"})
                fig_atm_s.update_traces(textposition='outside', textfont_color='white') # Texto blanco y fuera
                fig_atm_s.update_layout(height=GRAPH_HEIGHT, margin=dict(t=60, b=40, l=40, r=20), 
                                        yaxis_title="Azufre (ppm)",
                                        yaxis_range=[0, max_y_val_s * Y_AXIS_PADDING_FACTOR])
                st.plotly_chart(fig_atm_s, use_container_width=True)
            else: st.caption("Sin datos de azufre atmosférico para graficar.")

        col_vac1, col_vac2 = st.columns(2)
        with col_vac1:
            if not vac_plot_df.empty: 
                y_col_vac_yield = "Rend. Vol (%)" 
                max_y_val_vac = vac_plot_df[y_col_vac_yield].max() if not vac_plot_df[y_col_vac_yield].empty else 0
                if max_y_val_vac == 0: max_y_val_vac = 1
                
                fig_vac_y = px.bar(vac_plot_df, x="Corte", y=y_col_vac_yield, 
                                   title="Rend. Vol. Vacío (s/Alim. Vacío)", 
                                   text_auto='.2f',
                                   labels={"Corte": "Producto de Vacío", y_col_vac_yield: "Rendimiento Vol. (%)"})
                fig_vac_y.update_traces(textposition='outside', textfont_color='white') # Texto blanco y fuera
                fig_vac_y.update_layout(height=GRAPH_HEIGHT, margin=dict(t=60, b=40, l=40, r=20), 
                                        yaxis_title="Rendimiento Vol. (%) s/Alim. Vacío",
                                        yaxis_range=[0, max_y_val_vac * Y_AXIS_PADDING_FACTOR])
                st.plotly_chart(fig_vac_y, use_container_width=True)
            else: st.caption("Sin datos de rendimiento de vacío para graficar.")
        
        with col_vac2:
            if not vac_plot_df.empty:
                y_col_vac_sulfur = "Azufre (ppm)"
                max_y_val_vac_s = vac_plot_df[y_col_vac_sulfur].max() if not vac_plot_df[y_col_vac_sulfur].empty else 0
                if max_y_val_vac_s == 0: max_y_val_vac_s = 10
                
                fig_vac_s = px.bar(vac_plot_df, x="Corte", y=y_col_vac_sulfur, 
                                   title="Azufre en Prod. Vacío (ppm)", 
                                   text_auto='.0f',
                                   labels={"Corte": "Producto de Vacío", y_col_vac_sulfur: "Azufre (ppm)"})
                fig_vac_s.update_traces(textposition='outside', textfont_color='white') # Texto blanco y fuera
                fig_vac_s.update_layout(height=GRAPH_HEIGHT, margin=dict(t=60, b=40, l=40, r=20), 
                                        yaxis_title="Azufre (ppm)",
                                        yaxis_range=[0, max_y_val_vac_s * Y_AXIS_PADDING_FACTOR])
                st.plotly_chart(fig_vac_s, use_container_width=True)
            else: st.caption("Sin datos de azufre de vacío para graficar.")

        save_refinery_scenario_ui(original_feed_processed, 
                                  st.session_state.get('calculated_atmospheric_distillates', []),
                                  st.session_state.get('calculated_vacuum_products', []),
                                  st.session_state.get('calculated_atmospheric_residue'),
                                  api_factor_val)
# --- Fin del Script ---
