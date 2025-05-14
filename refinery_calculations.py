# Archivo: refinery_calculations.py
import numpy as np
import pandas as pd # Importado por si se necesita en futuras expansiones
import logging
from typing import List, Dict, Any, Optional, Tuple

# Configuración de logging (idealmente en el script principal)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')

# --- Funciones de Utilidad a Nivel de Módulo ---
def api_to_sg(api: Optional[float]) -> Optional[float]:
    """Convierte Gravedad API a Gravedad Específica (SG)."""
    if api is None:
        return None
    try:
        return 141.5 / (float(api) + 131.5)
    except (ValueError, TypeError):
        # No se usa self.name aquí porque es una función a nivel de módulo
        logging.warning(f"Could not convert API value '{api}' to float for SG calculation.")
        return None

def sg_to_api(sg: Optional[float]) -> Optional[float]:
    """Convierte Gravedad Específica (SG) a Gravedad API."""
    if sg is None:
        return None
    try:
        sg_float = float(sg)
        if sg_float == 0:
            logging.warning("SG value is 0, cannot calculate API.")
            return None
        return (141.5 / sg_float) - 131.5
    except (ValueError, TypeError):
        # No se usa self.name aquí
        logging.warning(f"Could not convert SG value '{sg}' to float for API calculation.")
        return None

# --- Funciones de Conversión de Curvas de Destilación (Placeholders) ---
def placeholder_convert_d86_to_tbp(d86_data: List[Tuple[float, float]], api_gravity: Optional[float] = None) -> List[Tuple[float, float]]:
    logging.warning("ASTM D86 to TBP conversion is a placeholder. Implement actual correlation.")
    return d86_data

def placeholder_convert_d1160_to_tbp(d1160_data: List[Tuple[float, float]], api_gravity: Optional[float] = None) -> List[Tuple[float, float]]:
    logging.warning("ASTM D1160 to TBP conversion is a placeholder. Implement actual correlation.")
    return d1160_data

def placeholder_convert_d2887_to_tbp(d2887_data: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    logging.warning("ASTM D2887 (SimDis) to TBP conversion is a placeholder. Implement actual correlation.")
    return d2887_data

def placeholder_convert_d7169_to_tbp(d7169_data: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    logging.warning("ASTM D7169 (SimDis High Temp) to TBP conversion is a placeholder. Implement actual correlation.")
    return d7169_data

class CrudeOil:
    """
    Represents a crude oil or a blend with its basic properties and distillation curve.
    The internal distillation curve is always processed and stored as TBP equivalent.
    """
    def __init__(self, name: str, api_gravity: float, sulfur_content_wt_percent: float,
                 distillation_data_percent_vol_temp_C: list,
                 distillation_curve_type: str = "TBP",
                 is_blend: bool = False,
                 verbose: bool = False):
        
        self.name = name
        self.verbose = verbose

        if self.verbose:
            logging.info(f"CrudeOil INIT: Creating instance for '{self.name}'. Received distillation data type: {type(distillation_data_percent_vol_temp_C)}, first 5 points: {str(distillation_data_percent_vol_temp_C[:5]) if isinstance(distillation_data_percent_vol_temp_C, list) else 'Not a list'}")

        try:
            self.api_gravity = float(api_gravity)
        except (ValueError, TypeError) as e:
            logging.error(f"CrudeOil INIT ERROR for '{self.name}': API gravity ('{api_gravity}') is not a valid number. Error: {e}")
            raise ValueError(f"API gravity para '{self.name}' debe ser un número. Recibido: '{api_gravity}'. Error: {e}")
        
        try:
            self.sulfur_total_wt_percent = float(sulfur_content_wt_percent)
        except (ValueError, TypeError) as e:
            logging.error(f"CrudeOil INIT ERROR for '{self.name}': Sulfur content ('{sulfur_content_wt_percent}') is not a valid number. Error: {e}")
            raise ValueError(f"Contenido de azufre para '{self.name}' debe ser un número. Recibido: '{sulfur_content_wt_percent}'. Error: {e}")

        self.original_raw_distillation_data = distillation_data_percent_vol_temp_C
        self.original_distillation_curve_type = distillation_curve_type.upper()
        self.is_blend = is_blend
        
        tbp_equivalent_curve_data = self._convert_to_tbp_if_needed(
            distillation_data_percent_vol_temp_C,
            self.original_distillation_curve_type,
            self.api_gravity
        )
        
        self._process_and_validate_distillation_curve(tbp_equivalent_curve_data)

        self.sg = api_to_sg(self.api_gravity) # Usa la función a nivel de módulo
        
        if self.distillation_temperatures_C and len(self.distillation_temperatures_C) > 0:
            self.ibp_C = self.distillation_temperatures_C[0]
            self.fbp_C = self.distillation_temperatures_C[-1]
            try:
                self.t50_C = self.get_temp_at_volume(50.0)
            except ValueError: # Handle case where get_temp_at_volume might fail (e.g. curve too short)
                self.t50_C = (self.ibp_C + self.fbp_C) / 2 if self.ibp_C is not None and self.fbp_C is not None else None # Fallback simple
                if self.verbose: logging.warning(f"Could not interpolate T50 for {self.name}, using midpoint as fallback.")
        else:
            self.ibp_C = None
            self.fbp_C = None
            self.t50_C = None
            if self.verbose:
                logging.warning(f"CrudeOil INIT WARN for '{self.name}': Distillation curve empty after processing. IBP/FBP/T50 could not be determined.")

        if self.verbose:
            ibp_display = f"{self.ibp_C:.1f}" if self.ibp_C is not None else "N/A"
            fbp_display = f"{self.fbp_C:.1f}" if self.fbp_C is not None else "N/A"
            t50_display = f"{self.t50_C:.1f}" if self.t50_C is not None else "N/A"
            num_points = len(self.distillation_volumes_percent) if self.distillation_volumes_percent is not None else 0
            
            logging.info(f"CrudeOil INIT: Instance for '{self.name}' created. API: {self.api_gravity:.1f}, S: {self.sulfur_total_wt_percent:.2f}%")
            logging.info(f"  Processed TBP IBP: {ibp_display}°C, FBP: {fbp_display}°C, T50: {t50_display}°C ({num_points} points)")

    def _convert_to_tbp_if_needed(self, original_curve_data: list, curve_type: str, api_gravity_for_conversion: Optional[float]) -> List[Tuple[float, float]]:
        valid_data_for_conversion = []
        if not isinstance(original_curve_data, list):
            if isinstance(original_curve_data, pd.DataFrame) and "Volumen (%)" in original_curve_data and "Temperatura (°C)" in original_curve_data:
                logging.warning(f"Attempting to convert DataFrame to list of tuples for {self.name} in _convert_to_tbp_if_needed")
                try:
                    vols = pd.to_numeric(original_curve_data["Volumen (%)"], errors='coerce')
                    temps = pd.to_numeric(original_curve_data["Temperatura (°C)"], errors='coerce')
                    valid_indices = vols.notna() & temps.notna()
                    original_curve_data = list(zip(vols[valid_indices], temps[valid_indices]))
                except Exception as e:
                    logging.error(f"Failed to convert DataFrame to list of tuples for {self.name} in _convert_to_tbp_if_needed: {e}")
                    raise ValueError(f"Invalid distillation data format for {self.name} (DataFrame conversion failed)")
            else:
                logging.error(f"Invalid format for original_curve_data for {self.name} before conversion. Expected list, got {type(original_curve_data)}.")
                raise ValueError(f"Invalid distillation data format for {self.name}")

        for i, p in enumerate(original_curve_data):
            if not (isinstance(p, (list, tuple)) and len(p) == 2):
                logging.error(f"Point {i} in original_curve_data for {self.name} is not a pair: {p}")
                raise ValueError(f"Invalid point format in distillation data for {self.name}")
            try:
                vol = float(p[0])
                temp = float(p[1])
                if pd.isna(vol) or pd.isna(temp):
                     logging.warning(f"NaN found in point {i} for {self.name}: ({p[0]}, {p[1]}). Skipping point.")
                     continue
                valid_data_for_conversion.append((vol, temp))
            except (ValueError, TypeError):
                logging.error(f"Non-numeric data in point {i} for {self.name}: {p}")
                raise ValueError(f"Non-numeric data in distillation point for {self.name}")
        
        if not valid_data_for_conversion and original_curve_data: 
            logging.error(f"No valid numeric points found in original_curve_data for {self.name} after filtering NaNs/errors.")
            raise ValueError(f"No valid numeric points in distillation data for {self.name}")

        if curve_type == "TBP":
            return valid_data_for_conversion
        elif curve_type == "ASTM D86":
            return placeholder_convert_d86_to_tbp(valid_data_for_conversion, api_gravity_for_conversion)
        elif curve_type == "ASTM D1160":
            return placeholder_convert_d1160_to_tbp(valid_data_for_conversion, api_gravity_for_conversion)
        elif curve_type == "ASTM D2887":
            return placeholder_convert_d2887_to_tbp(valid_data_for_conversion)
        elif curve_type == "ASTM D7169":
            return placeholder_convert_d7169_to_tbp(valid_data_for_conversion)
        
        logging.warning(f"Unsupported/placeholder distillation curve type '{curve_type}' for conversion for crude '{self.name}'. Using data as is.")
        return valid_data_for_conversion

    def _process_and_validate_distillation_curve(self, dist_data: list):
        if self.verbose:
            logging.info(f"_process_and_validate_distillation_curve for '{self.name}': Received type {type(dist_data)}. Data (first 5): {str(dist_data[:5]) if isinstance(dist_data, list) else 'Not a list'}")

        if not isinstance(dist_data, list):
            logging.error(f"_process_and_validate_distillation_curve ERROR for '{self.name}': Expected a list of points, got {type(dist_data)}.")
            raise ValueError(f"Formato/datos no numéricos en curva de destilación para '{self.name}': Se esperaba una lista de puntos, se obtuvo {type(dist_data)}.")

        processed_points = []
        if not dist_data:
            logging.warning(f"_process_and_validate_distillation_curve WARN for '{self.name}': Input distillation data list is empty.")
        
        for i, point in enumerate(dist_data):
            if self.verbose:
                logging.debug(f"  Processing point {i+1}/{len(dist_data)}: '{str(point)}' (type: {type(point)})")
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                logging.error(f"_process_and_validate_distillation_curve ERROR for '{self.name}': Point {i+1} ('{str(point)}') is not tuple/list of 2 elements.")
                raise ValueError(f"Formato/datos no numéricos en curva de destilación para '{self.name}': El punto {i+1} ('{str(point)}') no es una tupla/lista de 2 elementos.")
            try:
                vol_percent = float(point[0])
                temp_c = float(point[1])
                if pd.isna(vol_percent) or pd.isna(temp_c): 
                    raise ValueError("Contiene NaN")
            except (ValueError, TypeError) as e:
                logging.error(f"_process_and_validate_distillation_curve ERROR for '{self.name}': Point {i+1} ('{str(point)}') contains non-numeric values or NaN. Error: {e}")
                raise ValueError(f"Formato/datos no numéricos en curva de destilación para '{self.name}': El punto {i+1} ('{str(point)}') contiene valores no numéricos o NaN. Error: {e}")
            
            if not (0 <= vol_percent <= 100):
                 if self.verbose: 
                    logging.warning(f"  Warning for '{self.name}': Distillation point {point} with volume ({vol_percent}%) outside [0,100]. It will be included.")
            
            processed_points.append((vol_percent, temp_c))

        if len(processed_points) < 2:
            logging.error(f"_process_and_validate_distillation_curve ERROR for '{self.name}': Less than 2 valid points ({len(processed_points)}). Points: {str(processed_points)}")
            self.distillation_volumes_percent = []
            self.distillation_temperatures_C = []
            raise ValueError(f"Curva de destilación para '{self.name}' tiene menos de 2 puntos válidos después del procesamiento. Se requieren al menos IBP y FBP. Puntos procesados: {str(processed_points)}")

        processed_points.sort(key=lambda x: x[0])
        
        unique_points_dict = {vol: temp for vol, temp in processed_points}
        unique_sorted_points = sorted(unique_points_dict.items())

        self.distillation_volumes_percent = [p[0] for p in unique_sorted_points]
        self.distillation_temperatures_C = [p[1] for p in unique_sorted_points]

        if 0.0 not in self.distillation_volumes_percent:
            logging.error(f"_process_and_validate_distillation_curve ERROR for '{self.name}': Curve does not include 0% vol (IBP). Volumes: {str(self.distillation_volumes_percent)}")
            raise ValueError(f"Curva de destilación para '{self.name}' debe incluir un punto para 0% vol (IBP). Puntos de volumen: {str(self.distillation_volumes_percent)}")
        
        if 100.0 not in self.distillation_volumes_percent:
            logging.warning(f"_process_and_validate_distillation_curve WARN for '{self.name}': Curve does not include 100% vol (FBP). Volumes: {str(self.distillation_volumes_percent)}. Last point will be considered FBP.")

        if self.verbose:
            logging.info(f"_process_and_validate_distillation_curve for '{self.name}': Processing successful. {len(self.distillation_volumes_percent)} unique valid points.")

    def get_temp_at_volume(self, target_volume_percent: float) -> float:
        if not self.distillation_volumes_percent or not self.distillation_temperatures_C or len(self.distillation_volumes_percent) < 2:
            logging.error(f"get_temp_at_volume ERROR for '{self.name}': Distillation curve not available or incomplete ({len(self.distillation_volumes_percent)} points) for interpolation at {target_volume_percent}%.")
            raise ValueError(f"Curva de destilación no disponible o incompleta para '{self.name}' para interpolar.")
        
        vol_array = np.array(self.distillation_volumes_percent)
        temp_array = np.array(self.distillation_temperatures_C)
        
        if self.verbose and (target_volume_percent < vol_array[0] or target_volume_percent > vol_array[-1]):
            logging.warning(f"  Warning for '{self.name}' in get_temp_at_volume: Extrapolating temperature for {target_volume_percent}%. Curve range: [{vol_array[0]}%, {vol_array[-1]}%]")
        
        return float(np.interp(target_volume_percent, vol_array, temp_array))

    def get_volume_at_temp(self, target_temp_C: float) -> float:
        if not self.distillation_volumes_percent or not self.distillation_temperatures_C or len(self.distillation_volumes_percent) < 2:
            logging.error(f"get_volume_at_temp ERROR for '{self.name}': Distillation curve not available or incomplete for interpolation at {target_temp_C}°C.")
            raise ValueError(f"Curva de destilación no disponible o incompleta para '{self.name}' para interpolar.")

        vol_array = np.array(self.distillation_volumes_percent)
        temp_array = np.array(self.distillation_temperatures_C)

        if not np.all(np.diff(temp_array) >= 0): 
            if self.verbose:
                logging.warning(f"  Warning for '{self.name}' in get_volume_at_temp: Temperature array is not strictly monotonically increasing. Interpolation might be affected.")
        
        min_temp, max_temp = temp_array[0], temp_array[-1]
        if target_temp_C < min_temp:
            if self.verbose: logging.warning(f"  Target temp {target_temp_C}°C is below IBP {min_temp}°C for '{self.name}'. Result will be 0%.")
            return 0.0
        if target_temp_C > max_temp:
            if self.verbose: logging.warning(f"  Target temp {target_temp_C}°C is above FBP {max_temp}°C for '{self.name}'. Result will be 100%.")
            return 100.0
            
        return float(np.interp(target_temp_C, temp_array, vol_array))

    def __repr__(self):
        sg_display = f"{self.sg:.4f}" if self.sg is not None else "N/A"
        ibp_repr = f"{self.ibp_C:.1f}" if self.ibp_C is not None else "N/A"
        fbp_repr = f"{self.fbp_C:.1f}" if self.fbp_C is not None else "N/A"
        t50_repr = f"{self.t50_C:.1f}" if self.t50_C is not None else "N/A"
        
        return (f"CrudeOil('{self.name}', API={self.api_gravity:.1f}, SG={sg_display}, S={self.sulfur_total_wt_percent:.2f}%, "
                f"OrigCurveType: {self.original_distillation_curve_type}, TBP_Pts={len(self.distillation_volumes_percent) if self.distillation_volumes_percent else 0}, "
                f"IBP={ibp_repr}, FBP={fbp_repr}, T50={t50_repr})")


def create_blend_from_crudes(components_data_list: List[Dict[str, Any]], verbose: bool = False) -> CrudeOil:
    if not components_data_list:
        logging.error("create_blend_from_crudes ERROR: The component list is empty.")
        raise ValueError("La lista de componentes no puede estar vacía para crear una mezcla.")
    
    if verbose:
        logging.info(f"create_blend_from_crudes: Starting blend creation with {len(components_data_list)} components.")

    actual_total_proportion = 0.0
    blended_api_sum_product = 0.0
    blended_sulfur_sum_product = 0.0
    component_crude_objects_for_blend = []

    for i, comp_data in enumerate(components_data_list):
        comp_name_for_log = str(comp_data.get('name', f'Unknown Comp {i+1}'))
        if verbose:
            logging.info(f"  Processing component {i+1}/{len(components_data_list)} for blend: '{comp_name_for_log}'")

        try:
            proportion = float(comp_data.get('proportion_vol', 0.0))
        except (ValueError, TypeError):
            logging.error(f"  Invalid proportion for component '{comp_name_for_log}'. Skipping.")
            continue

        if proportion <= 1e-6:
            if verbose: logging.info(f"    Component '{comp_name_for_log}' skipped (proportion <= 0).")
            continue
        
        current_dist_data = comp_data.get('distillation_data', [])
        if verbose: 
            logging.info(f"    For '{comp_name_for_log}': Type of distillation_data received: {type(current_dist_data)}")
            if isinstance(current_dist_data, list):
                logging.info(f"    For '{comp_name_for_log}': distillation_data (first 5 points): {str(current_dist_data[:5])}")
                for idx, point_data in enumerate(current_dist_data[:5]):
                    logging.info(f"      Point {idx}: '{str(point_data)}', Type: {type(point_data)}")
                    if isinstance(point_data, (list, tuple)) and len(point_data) == 2:
                        val1_type = type(point_data[0]); val2_type = type(point_data[1])
                        logging.info(f"        Val1: '{str(point_data[0])}' (Type {val1_type}), Val2: '{str(point_data[1])}' (Type {val2_type})")
                    elif isinstance(point_data, (list, tuple)):
                         logging.warning(f"        Point {idx} is tuple/list but not 2 elements: length {len(point_data)}")
            else:
                logging.warning(f"    For '{comp_name_for_log}': distillation_data IS NOT A LIST.")

        try:
            crude_comp_obj = CrudeOil(
                name=comp_name_for_log,
                api_gravity=float(comp_data.get('api')),
                sulfur_content_wt_percent=float(comp_data.get('sulfur')),
                distillation_data_percent_vol_temp_C=current_dist_data,
                distillation_curve_type=str(comp_data.get('distillation_curve_type', 'TBP')),
                verbose=verbose
            )
            component_crude_objects_for_blend.append({'obj': crude_comp_obj, 'proportion_vol': proportion})
            blended_api_sum_product += crude_comp_obj.api_gravity * proportion
            blended_sulfur_sum_product += crude_comp_obj.sulfur_total_wt_percent * proportion
            actual_total_proportion += proportion
        except ValueError as e:
            logging.error(f"create_blend_from_crudes ERROR: Failed to process component '{comp_name_for_log}' for blend. Error: {e}")
            raise ValueError(f"Error al procesar componente '{comp_name_for_log}' para la mezcla: {e}")
        except Exception as e_generic:
            logging.error(f"create_blend_from_crudes UNEXPECTED ERROR: Failed to process component '{comp_name_for_log}'. Error: {e_generic}")
            raise Exception(f"Error inesperado al procesar componente '{comp_name_for_log}' para la mezcla: {e_generic}")

    if actual_total_proportion <= 1e-6:
        logging.error("create_blend_from_crudes ERROR: Total proportion of valid components is zero.")
        raise ValueError("La proporción total de los componentes válidos es cero. No se puede calcular la mezcla.")

    if not np.isclose(actual_total_proportion, 100.0, atol=0.1): 
        if verbose:
            logging.warning(f"  Sum of valid component proportions is {actual_total_proportion}%. API/Sulfur will be based on this sum.")
    
    final_blended_api = blended_api_sum_product / actual_total_proportion
    final_blended_sulfur = blended_sulfur_sum_product / actual_total_proportion

    component_display_names = [c['obj'].name for c in component_crude_objects_for_blend]
    final_blended_name = "Mezcla de " + ", ".join(component_display_names)
    
    final_blended_distillation_data = []
    final_blended_curve_type = "TBP"

    if component_crude_objects_for_blend:
        all_vol_points_set = set()
        for comp_item in component_crude_objects_for_blend:
            if comp_item['obj'].distillation_volumes_percent:
                all_vol_points_set.update(comp_item['obj'].distillation_volumes_percent)
        all_vol_points = sorted(list(all_vol_points_set))
        
        if not all_vol_points:
             final_blended_distillation_data = [(0.0, 60.0), (100.0, 500.0)]
             if verbose: logging.warning("  Warning: No component had a valid distillation curve; using generic default curve for the blend.")
        else:
            if 0.0 not in all_vol_points: all_vol_points.insert(0, 0.0)
            if 100.0 not in all_vol_points: all_vol_points.append(100.0)
            all_vol_points = sorted(list(set(all_vol_points)))

            for vol_p in all_vol_points:
                weighted_temp_sum = 0.0
                sum_of_proportions_at_this_vol = 0.0
                for comp_item in component_crude_objects_for_blend:
                    comp_obj = comp_item['obj']
                    comp_prop_normalized = (comp_item['proportion_vol'] / actual_total_proportion) if actual_total_proportion > 0 else 0
                    try:
                        temp_at_vol = comp_obj.get_temp_at_volume(vol_p)
                        weighted_temp_sum += temp_at_vol * comp_prop_normalized
                        sum_of_proportions_at_this_vol += comp_prop_normalized
                    except ValueError:
                        if verbose: logging.debug(f"    Debug: Could not get temp for {vol_p}% in {comp_obj.name} during blend curve calculation (likely out of range for this comp).")
                        pass
                
                if sum_of_proportions_at_this_vol > 1e-6:
                    avg_temp_at_vol = weighted_temp_sum / sum_of_proportions_at_this_vol
                    final_blended_distillation_data.append((vol_p, avg_temp_at_vol))
                elif verbose:
                     logging.debug(f"    Debug: Could not calculate blended temp for {vol_p}% (no significant component contribution).")
            
            if len(final_blended_distillation_data) < 2:
                if verbose: logging.warning(f"  Warning: Blended distillation curve resulted in less than 2 points ({str(final_blended_distillation_data)}). Using generic default curve.")
                final_blended_distillation_data = [(0.0, 60.0), (100.0, 500.0)]
    else:
        final_blended_distillation_data = [(0.0, 60.0), (100.0, 500.0)]
        if verbose: logging.warning("  Warning: No valid components for blend; using generic default curve for the blend.")

    if final_blended_distillation_data and final_blended_distillation_data[0][0] != 0.0:
        if final_blended_distillation_data[0][0] > 0.0 and len(final_blended_distillation_data) >=1 : 
            logging.warning(f"  Blended curve missing 0% IBP. Prepending with first point's temperature. First point: {final_blended_distillation_data[0]}")
            final_blended_distillation_data.insert(0, (0.0, final_blended_distillation_data[0][1]))
        elif not final_blended_distillation_data : 
             logging.warning(f"  Blended curve is empty and missing 0% IBP. Setting to default.")
             final_blended_distillation_data = [(0.0, 60.0), (100.0, 500.0)]

    blend_properties_dict = {
        'name': final_blended_name,
        'api_gravity': final_blended_api,
        'sulfur_content_wt_percent': final_blended_sulfur,
        'distillation_data_percent_vol_temp_C': final_blended_distillation_data,
        'distillation_curve_type': final_blended_curve_type,
        'is_blend': True,
        'verbose': verbose
    }
    if verbose:
        logging.info(f"create_blend_from_crudes: Creating final CrudeOil object for the blend '{final_blended_name}'.")
    
    blended_crude_oil_object = CrudeOil(**blend_properties_dict)

    if verbose:
        logging.info(f"create_blend_from_crudes: Blend '{blended_crude_oil_object.name}' created successfully. API: {blended_crude_oil_object.api_gravity:.1f}")
    return blended_crude_oil_object

class DistillationCut:
    def __init__(self, name: str, t_initial_C: float, t_final_C: float, 
                 crude_oil_feed: CrudeOil, 
                 api_sensitivity_factor: float = 7.0, 
                 verbose: bool = False):
        self.name = name
        self.t_initial_C = float(t_initial_C)
        self.t_final_C = float(t_final_C)
        self.crude_oil_feed = crude_oil_feed 
        self.api_sensitivity_factor = api_sensitivity_factor
        self.verbose = verbose

        self.yield_vol_percent: Optional[float] = None
        self.vabp_C: Optional[float] = None 
        self.api_cut: Optional[float] = None
        self.sg_cut: Optional[float] = None
        self.yield_wt_percent: Optional[float] = None
        self.sulfur_cut_wt_percent: Optional[float] = None
        self.sulfur_cut_ppm: Optional[float] = None
        self.is_gas_cut: bool = name.lower().startswith("gas") and self.t_final_C <= 40 

        self._calculate_properties()

    def _calculate_properties(self):
        if self.crude_oil_feed.distillation_volumes_percent is None or not self.crude_oil_feed.distillation_volumes_percent:
            if self.verbose: logging.warning(f"Cannot calculate properties for cut '{self.name}', feed crude '{self.crude_oil_feed.name}' has no distillation curve.")
            return

        vol_at_t_initial = self.crude_oil_feed.get_volume_at_temp(self.t_initial_C)
        vol_at_t_final = self.crude_oil_feed.get_volume_at_temp(self.t_final_C)
        self.yield_vol_percent = max(0.0, vol_at_t_final - vol_at_t_initial)

        if self.yield_vol_percent < 1e-6: 
            if self.verbose: logging.info(f"Cut '{self.name}' has ~zero yield ({self.yield_vol_percent:.4f}%). Properties will be None or default.")
            self.api_cut = None 
            self.sg_cut = None
            self.sulfur_cut_wt_percent = 0.0 
            self.sulfur_cut_ppm = 0.0
            self.yield_wt_percent = 0.0
            self.vabp_C = (self.t_initial_C + self.t_final_C) / 2.0 
            return

        vol_mid_point_of_cut = vol_at_t_initial + self.yield_vol_percent / 2.0
        self.vabp_C = self.crude_oil_feed.get_temp_at_volume(vol_mid_point_of_cut)
        
        if self.crude_oil_feed.t50_C is not None and self.vabp_C is not None and self.crude_oil_feed.api_gravity is not None:
            api_deviation = (self.crude_oil_feed.t50_C - self.vabp_C) / self.api_sensitivity_factor 
            self.api_cut = self.crude_oil_feed.api_gravity + api_deviation
            self.api_cut = max(0.0, min(100.0, self.api_cut)) if self.api_cut is not None else None
        else: 
            self.api_cut = self.crude_oil_feed.api_gravity 

        self.sg_cut = api_to_sg(self.api_cut) # Usa la función a nivel de módulo

        if self.sg_cut is not None and self.crude_oil_feed.sg is not None and self.crude_oil_feed.sg > 1e-6:
            density_correction = 0.85 if self.is_gas_cut else 1.0 
            self.yield_wt_percent = self.yield_vol_percent * (self.sg_cut / self.crude_oil_feed.sg) * density_correction
        else:
            self.yield_wt_percent = self.yield_vol_percent 

        if self.api_cut is not None and self.crude_oil_feed.api_gravity is not None and self.crude_oil_feed.sulfur_total_wt_percent is not None:
            sulfur_ratio_factor = (self.crude_oil_feed.api_gravity / self.api_cut) if self.api_cut > 1e-6 else 1.0
            sulfur_ratio_factor = max(0.1, min(3.0, sulfur_ratio_factor)) 
            estimated_sulfur = self.crude_oil_feed.sulfur_total_wt_percent * sulfur_ratio_factor
            self.sulfur_cut_wt_percent = max(0.00001, min(estimated_sulfur, 10.0)) 
        else:
            self.sulfur_cut_wt_percent = self.crude_oil_feed.sulfur_total_wt_percent 

        self.sulfur_cut_ppm = (self.sulfur_cut_wt_percent * 10000) if self.sulfur_cut_wt_percent is not None else None
        
        if self.verbose:
            api_display = f"{self.api_cut:.1f}" if self.api_cut is not None else "N/A"
            sulfur_display = f"{self.sulfur_cut_wt_percent:.4f}" if self.sulfur_cut_wt_percent is not None else "N/A"
            yield_display = f"{self.yield_vol_percent:.2f}" if self.yield_vol_percent is not None else "N/A"
            logging.info(f"Cut '{self.name}' calculated: T_init={self.t_initial_C:.1f}°C, T_final={self.t_final_C:.1f}°C, YieldVol={yield_display}%, API={api_display}, S_wt%={sulfur_display}")


    def to_dict(self) -> Dict[str, Any]:
        return {
            "Corte": self.name,
            "T Inicial (°C)": self.t_initial_C,
            "T Final (°C)": self.t_final_C,
            "Rend. Vol (%)": self.yield_vol_percent,
            "Rend. Peso (%)": self.yield_wt_percent,
            "API Corte": self.api_cut,
            "SG Corte": self.sg_cut,
            "Azufre (%peso)": self.sulfur_cut_wt_percent,
            "Azufre (ppm)": self.sulfur_cut_ppm,
            "VABP (°C)": self.vabp_C
        }

def calculate_atmospheric_cuts(
    crude_oil_feed: CrudeOil, 
    atmospheric_cut_definitions: List[Tuple[str, float]], 
    verbose: bool = False,
    api_sensitivity_factor: float = 7.0,
    empirical_data_for_crude: Optional[Dict[str,Any]] = None 
) -> Tuple[List[DistillationCut], Optional[DistillationCut]]:
    if verbose:
        logging.info(f"Calculating atmospheric cuts for: {crude_oil_feed.name}")
        logging.info(f"Atmospheric cut definitions: {atmospheric_cut_definitions}")

    distillate_cuts: List[DistillationCut] = []
    current_t_initial = crude_oil_feed.ibp_C if crude_oil_feed.ibp_C is not None else 0.0
    if verbose:
            logging.info(f"Initial T_initial for atmospheric cuts (from crude IBP): {current_t_initial:.1f}°C")


    sorted_cut_definitions = sorted(atmospheric_cut_definitions, key=lambda x: x[1])

    for cut_name, t_final_C in sorted_cut_definitions:
        if t_final_C <= current_t_initial: 
            if verbose: logging.warning(f"Skipping atmospheric cut '{cut_name}' as its T_final ({t_final_C}°C) is not greater than current T_initial ({current_t_initial}°C).")
            continue
        
        if verbose:
            logging.info(f"Defining atmospheric cut '{cut_name}': T_initial={current_t_initial:.1f}°C, T_final={t_final_C:.1f}°C")
        cut = DistillationCut(name=cut_name, 
                              t_initial_C=current_t_initial, 
                              t_final_C=t_final_C, 
                              crude_oil_feed=crude_oil_feed,
                              api_sensitivity_factor=api_sensitivity_factor,
                              verbose=verbose)
        distillate_cuts.append(cut)
        current_t_initial = t_final_C # This is the FBP of the current cut, and IBP for the next
        if verbose:
            logging.info(f"  Next T_initial set to: {current_t_initial:.1f}°C (from FBP of '{cut_name}')")


    atmospheric_residue_obj: Optional[DistillationCut] = None
    if crude_oil_feed.fbp_C is not None and current_t_initial < crude_oil_feed.fbp_C:
        if verbose:
            logging.info(f"Defining Atmospheric Residue: T_initial={current_t_initial:.1f}°C, T_final={crude_oil_feed.fbp_C:.1f}°C")
        atmospheric_residue_obj = DistillationCut(name="Residuo Atmosférico",
                                                  t_initial_C=current_t_initial, # Starts where the last distillate ended
                                                  t_final_C=crude_oil_feed.fbp_C, 
                                                  crude_oil_feed=crude_oil_feed,
                                                  api_sensitivity_factor=api_sensitivity_factor,
                                                  verbose=verbose)
        if verbose and atmospheric_residue_obj and atmospheric_residue_obj.yield_vol_percent is not None: 
             logging.info(f"Atmospheric Residue calculated: YieldVol={atmospheric_residue_obj.yield_vol_percent:.2f}%, T_initial={atmospheric_residue_obj.t_initial_C:.1f}°C")
    elif verbose:
        logging.info(f"No significant atmospheric residue to calculate. Current T_initial: {current_t_initial}°C, Crude FBP: {crude_oil_feed.fbp_C}°C")

    if empirical_data_for_crude and empirical_data_for_crude.get("distribution_data", {}).get("cuts"):
        logging.warning("Empirical data application for atmospheric cuts is not fully implemented yet.")

    return distillate_cuts, atmospheric_residue_obj

def create_vacuum_feed_from_residue(
    original_crude_feed: CrudeOil, 
    atmospheric_residue: DistillationCut, 
    verbose: bool = False
) -> Optional[CrudeOil]:
    if atmospheric_residue.yield_vol_percent is None or atmospheric_residue.yield_vol_percent < 1e-3: 
        if verbose: logging.info("Atmospheric residue yield is too low to create vacuum feed.")
        return None

    if verbose:
        logging.info(f"Creating vacuum feed from atmospheric residue of '{original_crude_feed.name}'.")
        api_cut_display = f"{atmospheric_residue.api_cut:.1f}" if atmospheric_residue.api_cut is not None else "N/A"
        sulfur_cut_display = f"{atmospheric_residue.sulfur_cut_wt_percent:.4f}" if atmospheric_residue.sulfur_cut_wt_percent is not None else "N/A"
        logging.info(f"Atmospheric Residue props: T_initial={atmospheric_residue.t_initial_C:.1f}°C, API={api_cut_display}, S%wt={sulfur_cut_display}")
    
    # The T_initial for the vacuum feed *is* the T_initial of the atmospheric residue object
    # This temperature was the T_final of the last atmospheric distillate.
    vac_feed_t_initial = atmospheric_residue.t_initial_C
    vac_feed_t_final = original_crude_feed.fbp_C 

    if vac_feed_t_final is None or vac_feed_t_initial >= vac_feed_t_final:
        if verbose: logging.warning(f"Cannot create vacuum feed: T_initial_residue ({vac_feed_t_initial}) >= FBP_crude ({vac_feed_t_final}).")
        return None

    original_vols = np.array(original_crude_feed.distillation_volumes_percent)
    original_temps = np.array(original_crude_feed.distillation_temperatures_C)

    # We need to ensure the vacuum feed distillation curve starts at 0% volume at vac_feed_t_initial
    # and ends at 100% volume at vac_feed_t_final on its own scale.
    
    # Points from original curve that fall within the residue's temperature range
    mask = (original_temps >= vac_feed_t_initial) & (original_temps <= vac_feed_t_final)
    residue_temps_on_orig_curve = original_temps[mask]
    residue_vols_on_orig_curve = original_vols[mask]

    vac_feed_dist_curve_points = []

    # Ensure the first point of the vacuum feed curve is (0.0, vac_feed_t_initial)
    # We get the volume on the original crude curve corresponding to vac_feed_t_initial
    vol_at_vac_feed_t_initial = original_crude_feed.get_volume_at_temp(vac_feed_t_initial)
    vac_feed_dist_curve_points.append((vol_at_vac_feed_t_initial, vac_feed_t_initial))
    
    # Add points from the original curve that are strictly within the new IBP and FBP
    for vol, temp in zip(residue_vols_on_orig_curve, residue_temps_on_orig_curve):
        if temp > vac_feed_t_initial and temp < vac_feed_t_final: # Strictly between
            vac_feed_dist_curve_points.append((vol, temp))

    # Ensure the last point corresponds to vac_feed_t_final
    # We get the volume on the original crude curve corresponding to vac_feed_t_final
    vol_at_vac_feed_t_final = original_crude_feed.get_volume_at_temp(vac_feed_t_final)
    # Add this point only if it's different from the last added point to avoid duplicate temperatures if FBP is same as a point in residue_temps
    if not vac_feed_dist_curve_points or vac_feed_dist_curve_points[-1][1] < vac_feed_t_final:
         vac_feed_dist_curve_points.append((vol_at_vac_feed_t_final, vac_feed_t_final))


    # Remove duplicates by volume, then sort by volume before renormalization
    # This step handles cases where original curve might not be dense enough
    # or where prepended/appended points create duplicates.
    if vac_feed_dist_curve_points:
        temp_df_for_vac_feed = pd.DataFrame(vac_feed_dist_curve_points, columns=['vol_orig', 'temp'])
        temp_df_for_vac_feed.sort_values(by=['vol_orig', 'temp'], inplace=True) 
        temp_df_for_vac_feed.drop_duplicates(subset=['vol_orig'], keep='first', inplace=True) 
        temp_df_for_vac_feed.drop_duplicates(subset=['temp'], keep='first', inplace=True) # Ensure unique temperatures for interpolation
        
        unique_residue_vols_on_orig_curve = temp_df_for_vac_feed['vol_orig'].values
        unique_residue_temps_on_orig_curve = temp_df_for_vac_feed['temp'].values
    else: # Fallback if no points were gathered
        unique_residue_vols_on_orig_curve = np.array([vol_at_vac_feed_t_initial, vol_at_vac_feed_t_final])
        unique_residue_temps_on_orig_curve = np.array([vac_feed_t_initial, vac_feed_t_final])


    if len(unique_residue_vols_on_orig_curve) < 2:
        if verbose: logging.warning(f"Not enough unique points from original crude curve to define vacuum feed curve after filtering. Points found: {len(unique_residue_vols_on_orig_curve)}. Using simple 2-point curve.")
        vac_feed_dist_curve = [(0.0, vac_feed_t_initial), (100.0, vac_feed_t_final)]
    else:
        # Renormalize volumes for the vacuum feed's own curve (0% to 100%)
        vol_start_of_residue_on_orig_crude_for_renorm = unique_residue_vols_on_orig_curve[0]
        vol_end_of_residue_on_orig_crude_for_renorm = unique_residue_vols_on_orig_curve[-1]
        
        total_vol_span_of_residue_on_orig_crude = vol_end_of_residue_on_orig_crude_for_renorm - vol_start_of_residue_on_orig_crude_for_renorm
        
        if total_vol_span_of_residue_on_orig_crude < 1e-3: 
             if verbose: logging.warning("Residue volume span on original crude is too small for renormalization. Using simple 2-point curve for vacuum feed.")
             vac_feed_dist_curve = [(0.0, vac_feed_t_initial), (100.0, vac_feed_t_final)]
        else:
            renormalized_vols = ((unique_residue_vols_on_orig_curve - vol_start_of_residue_on_orig_crude_for_renorm) / total_vol_span_of_residue_on_orig_crude) * 100.0
            
            # Ensure the renormalized curve strictly starts at 0% and ends at 100%
            renormalized_vols[0] = 0.0
            renormalized_vols[-1] = 100.0
            
            # The temperatures correspond directly to the unique_residue_temps_on_orig_curve
            final_temps_for_vac_feed_curve = unique_residue_temps_on_orig_curve.copy()
            # Ensure first and last temperatures are exactly the vac_feed_t_initial and vac_feed_t_final
            final_temps_for_vac_feed_curve[0] = vac_feed_t_initial
            final_temps_for_vac_feed_curve[-1] = vac_feed_t_final


            vac_feed_dist_curve = list(zip(renormalized_vols, final_temps_for_vac_feed_curve))
            
            # Final check for duplicates in renormalized curve points (by volume) and ensure monotonicity
            temp_df = pd.DataFrame(vac_feed_dist_curve, columns=['vol', 'temp'])
            temp_df.drop_duplicates(subset=['vol'], keep='first', inplace=True)
            temp_df.sort_values(by='vol', inplace=True) # Ensure sorted by volume
            # Check temperature monotonicity again after ensuring start/end temps
            if not np.all(np.diff(temp_df['temp'].values) >= 0):
                if verbose: logging.warning(f"Vacuum feed curve temperatures are not strictly monotonic after final adjustments for {original_crude_feed.name}. This might affect interpolation. Curve: {temp_df.to_dict('records')}")
                # Attempt to fix by keeping first temp for duplicated volumes if temps are not monotonic
                # This is a simple fix; more complex scenarios might need more robust handling
                temp_df.drop_duplicates(subset=['temp'], keep='first', inplace=True)
                temp_df.sort_values(by='vol', inplace=True)


            vac_feed_dist_curve = list(zip(temp_df['vol'], temp_df['temp']))


    vac_feed_api = atmospheric_residue.api_cut if atmospheric_residue.api_cut is not None else original_crude_feed.api_gravity 
    vac_feed_sulfur = atmospheric_residue.sulfur_cut_wt_percent if atmospheric_residue.sulfur_cut_wt_percent is not None else original_crude_feed.sulfur_total_wt_percent 

    vacuum_feed_crude = CrudeOil(name=f"Alimentación Vacío (de {original_crude_feed.name})",
                                 api_gravity=vac_feed_api,
                                 sulfur_content_wt_percent=vac_feed_sulfur,
                                 distillation_data_percent_vol_temp_C=vac_feed_dist_curve,
                                 distillation_curve_type="TBP", 
                                 is_blend=original_crude_feed.is_blend, 
                                 verbose=verbose)
    if verbose:
        logging.info(f"Vacuum feed created: {vacuum_feed_crude.name}, API: {vacuum_feed_crude.api_gravity:.1f}")
        logging.info(f"  Vacuum feed TBP curve (first 5 points): {vacuum_feed_crude.original_raw_distillation_data[:5]}")
        logging.info(f"  Vacuum feed IBP_C: {vacuum_feed_crude.ibp_C:.1f}, FBP_C: {vacuum_feed_crude.fbp_C:.1f}")
    return vacuum_feed_crude

def calculate_vacuum_cuts(
    vacuum_feed: CrudeOil, 
    vacuum_cut_definitions: List[Tuple[str, float]], 
    atmospheric_residue_initial_temp: float, # MODIFICACIÓN: Pasar T_inicial del residuo explícitamente
    verbose: bool = False,
    api_sensitivity_factor: float = 7.0
) -> List[DistillationCut]:
    if verbose:
        logging.info(f"Calculating vacuum cuts for: {vacuum_feed.name}")
        logging.info(f"Vacuum cut definitions (TBP eq. atm.): {vacuum_cut_definitions}")
        logging.info(f"Received Atmospheric Residue Initial Temp (for first vacuum cut IBP): {atmospheric_residue_initial_temp:.1f}°C")

    vacuum_distillates: List[DistillationCut] = []
    
    # MODIFICACIÓN CLAVE: El IBP del primer corte de vacío es el T_final del último
    # destilado atmosférico (que es el T_inicial del residuo atmosférico).
    current_t_initial_vac = atmospheric_residue_initial_temp
    
    if verbose:
        logging.info(f"Initial T_initial for vacuum cuts (from Atm. Residue IBP): {current_t_initial_vac:.1f}°C")
        if vacuum_feed.ibp_C is not None and not np.isclose(vacuum_feed.ibp_C, current_t_initial_vac, atol=0.5): # atol para pequeñas diferencias de flotantes
            logging.warning(f"  Mismatch: Vacuum Feed IBP ({vacuum_feed.ibp_C:.1f}°C) vs. "
                            f"Atm Residue Initial Temp ({current_t_initial_vac:.1f}°C). Using Atm Residue Initial Temp.")


    sorted_vac_cut_definitions = sorted(vacuum_cut_definitions, key=lambda x: x[1])

    for cut_name, t_final_C_eq_atm in sorted_vac_cut_definitions:
        if t_final_C_eq_atm <= current_t_initial_vac:
            if verbose: logging.warning(f"Skipping vacuum cut '{cut_name}' as T_final_eq_atm ({t_final_C_eq_atm}°C) is not > current T_initial_vac ({current_t_initial_vac}°C).")
            continue
        
        if verbose:
            logging.info(f"Defining vacuum cut '{cut_name}': T_initial={current_t_initial_vac:.1f}°C, T_final={t_final_C_eq_atm:.1f}°C")

        vac_cut = DistillationCut(name=cut_name,
                                  t_initial_C=current_t_initial_vac,
                                  t_final_C=t_final_C_eq_atm,
                                  crude_oil_feed=vacuum_feed, 
                                  api_sensitivity_factor=api_sensitivity_factor,
                                  verbose=verbose)
        vacuum_distillates.append(vac_cut)
        current_t_initial_vac = t_final_C_eq_atm # FBP del corte actual es IBP del siguiente
        if verbose:
            logging.info(f"  Next T_initial for vacuum set to: {current_t_initial_vac:.1f}°C (from FBP of '{cut_name}')")


    if vacuum_feed.fbp_C is not None and current_t_initial_vac < vacuum_feed.fbp_C:
        if verbose:
            logging.info(f"Defining Vacuum Residue: T_initial={current_t_initial_vac:.1f}°C, T_final={vacuum_feed.fbp_C:.1f}°C")
        vacuum_residue_obj = DistillationCut(name="Residuo de Vacío",
                                             t_initial_C=current_t_initial_vac,
                                             t_final_C=vacuum_feed.fbp_C, 
                                             crude_oil_feed=vacuum_feed,
                                             api_sensitivity_factor=api_sensitivity_factor,
                                             verbose=verbose)
        vacuum_distillates.append(vacuum_residue_obj) 
        if verbose and vacuum_residue_obj and vacuum_residue_obj.yield_vol_percent is not None: 
            logging.info(f"Vacuum Residue calculated: YieldVol={vacuum_residue_obj.yield_vol_percent:.2f}%")
    elif verbose:
        logging.info(f"No significant vacuum residue to calculate. Current T_initial_vac: {current_t_initial_vac}°C, VacFeed FBP: {vacuum_feed.fbp_C}°C")
        
    return vacuum_distillates

# --- Example Test Block (Uncomment to run this file directly for testing) ---
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s')

#     example_crude_data = [(0,50),(10,100),(30,200),(50,280),(70,380),(90,480),(100,550)]
#     test_crude = CrudeOil(name="Crudo Ejemplo Test", api_gravity=32.0, sulfur_content_wt_percent=1.5,
#                           distillation_data_percent_vol_temp_C=example_crude_data, verbose=True)
#     print(f"\nTest Crude created: {test_crude}")

#     atm_cuts_def = [("Nafta Ligera", 85.0), ("Nafta Pesada", 150.0), ("Kerosene", 250.0), ("Diesel Ligero", 350.0)]
#     print(f"\n--- Testing Atmospheric Distillation for {test_crude.name} ---")
#     atm_distillates, atm_residue = calculate_atmospheric_cuts(test_crude, atm_cuts_def, verbose=True, api_sensitivity_factor=7.0)
    
#     print("\nAtmospheric Distillates:")
#     for cut in atm_distillates:
#         api_disp = f"{cut.api_cut:.1f}" if cut.api_cut is not None else "N/A"
#         s_disp = f"{cut.sulfur_cut_wt_percent:.4f}" if cut.sulfur_cut_wt_percent is not None else "N/A"
#         y_disp = f"{cut.yield_vol_percent:.2f}" if cut.yield_vol_percent is not None else "N/A"
#         print(f"  {cut.name}: T_init={cut.t_initial_C:.1f}, T_final={cut.t_final_C:.1f}, YieldVol={y_disp}%, API={api_disp}, S_wt%={s_disp}")
#     if atm_residue:
#         api_disp = f"{atm_residue.api_cut:.1f}" if atm_residue.api_cut is not None else "N/A"
#         s_disp = f"{atm_residue.sulfur_cut_wt_percent:.4f}" if atm_residue.sulfur_cut_wt_percent is not None else "N/A"
#         y_disp = f"{atm_residue.yield_vol_percent:.2f}" if atm_residue.yield_vol_percent is not None else "N/A"
#         print(f"Atmospheric Residue: T_init={atm_residue.t_initial_C:.1f}, T_final={atm_residue.t_final_C:.1f}, YieldVol={y_disp}%, API={api_disp}, S_wt%={s_disp}")
#     else:
#         print("No Atmospheric Residue produced.")

#     if atm_residue and atm_residue.yield_vol_percent is not None and atm_residue.yield_vol_percent > 0.1 :
#         print(f"\n--- Testing Vacuum Distillation ---")
#         # Aseguramos que atm_residue.t_initial_C sea usado para el primer corte de vacío
#         atm_residue_start_temp_for_vacuum = atm_residue.t_initial_C 
        
#         vacuum_feed = create_vacuum_feed_from_residue(test_crude, atm_residue, verbose=True)
#         if vacuum_feed:
#             ibp_disp = f"{vacuum_feed.ibp_C:.1f}" if vacuum_feed.ibp_C is not None else "N/A"
#             fbp_disp = f"{vacuum_feed.fbp_C:.1f}" if vacuum_feed.fbp_C is not None else "N/A"
#             print(f"Vacuum Feed created: {vacuum_feed.name}, API: {vacuum_feed.api_gravity:.1f}, IBP: {ibp_disp}, FBP: {fbp_disp}")
#             vac_cuts_def = [("LVGO", 420.0), ("MVGO", 480.0), ("HVGO", 520.0)] 
            
#             vacuum_products = calculate_vacuum_cuts(
#                 vacuum_feed, 
#                 vac_cuts_def, 
#                 atmospheric_residue_initial_temp=atm_residue_start_temp_for_vacuum, # Pasa la T_inicial del residuo
#                 verbose=True, 
#                 api_sensitivity_factor=5.0
#             )
#             print("\nVacuum Products:")
#             for prod in vacuum_products:
#                 api_disp = f"{prod.api_cut:.1f}" if prod.api_cut is not None else "N/A"
#                 s_disp = f"{prod.sulfur_cut_wt_percent:.4f}" if prod.sulfur_cut_wt_percent is not None else "N/A"
#                 y_disp = f"{prod.yield_vol_percent:.2f}" if prod.yield_vol_percent is not None else "N/A"
#                 print(f"  {prod.name}: T_init={prod.t_initial_C:.1f}, T_final={prod.t_final_C:.1f}, YieldVol={y_disp}%, API={api_disp}, S_wt%={s_disp}")
#         else:
#             print("Could not create vacuum feed.")
#     else:
#         print("\nSkipping vacuum distillation test as atmospheric residue yield is too low.")

#     test_components_valid = [
#         {'name': 'Crudo Ligero Valido', 'api': 35.0, 'sulfur': 0.2, 'proportion_vol': 60.0,
#          'distillation_data': [(0.0,50.0), (10.0,100.0), (50.0,280.0), (90.0,450.0), (100.0,500.0)],
#          'distillation_curve_type': 'TBP'},
#         {'name': 'Crudo Pesado Valido', 'api': 22.0, 'sulfur': 1.5, 'proportion_vol': 40.0,
#          'distillation_data': [(0.0,100.0), (10.0,180.0), (50.0,350.0), (90.0,550.0), (100.0,600.0)],
#          'distillation_curve_type': 'TBP'}
#     ]
#     print("\n--- Testing Blending ---")
#     try:
#         blend_valid = create_blend_from_crudes(test_components_valid, verbose=True)
#         print(f"SUCCESS: Valid blend created: {blend_valid.name}, API: {blend_valid.api_gravity:.1f}")
#         print(f"  Blend TBP Curve (first 5 points): {list(zip(blend_valid.distillation_volumes_percent, blend_valid.distillation_temperatures_C))[:5]}")
#     except Exception as e:
#         print(f"ERROR in Blending Test: {e}")
#         logging.exception("Exception in Blending Test")

#     test_components_problematic = [
#         {'name': 'Escalante (Ejemplo)', 'api': 23.7, 'sulfur': 0.24, 'proportion_vol': 100.0, 
#          'distillation_data': [(0,160), (10,248), (30,345), (50,432), (70,530), (90,670), (95,720), (100,750)], 
#          'distillation_curve_type': 'TBP'},
#         {'name': 'CSV Test', 'api': 30.0, 'sulfur': 1.0, 'proportion_vol': 0.0, 
#          'distillation_data': [(0.0, 60.0), (10.0, 150.5), (50.0, 280.0), (100.0, 500.0)], 
#          'distillation_curve_type': 'TBP'}
#     ]
#     print("\n--- Testing Blending with Escalante ---")
#     try:
#         blend_escalante = create_blend_from_crudes(test_components_problematic, verbose=True)
#         print(f"SUCCESS: Escalante blend created: {blend_escalante.name}, API: {blend_escalante.api_gravity:.1f}")
#     except Exception as e:
#         print(f"ERROR in Escalante Blending Test: {e}")
#         logging.exception("Exception in Escalante Blending Test")
