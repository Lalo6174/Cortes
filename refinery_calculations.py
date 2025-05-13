import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Placeholder para las funciones de conversión de curvas de destilación
# DEBERÁS IMPLEMENTAR ESTAS FUNCIONES CON CORRELACIONES REALES
def placeholder_convert_d86_to_tbp(d86_data: List[Tuple[float, float]], api_gravity: Optional[float] = None) -> List[Tuple[float, float]]:
    logging.warning("ASTM D86 to TBP conversion is a placeholder. Implement actual correlation.")
    # Ejemplo: Simplemente devuelve los datos originales o una transformación muy básica.
    # Esto NO es una conversión real.
    # En una implementación real, se aplicarían ecuaciones de Edmister, API, etc.
    # Por ejemplo, un simple ajuste (NO USAR EN PRODUCCIÓN):
    # return [(vol, temp + 10) for vol, temp in d86_data] # Ejemplo de ajuste simple
    return d86_data

def placeholder_convert_d1160_to_tbp(d1160_data: List[Tuple[float, float]], api_gravity: Optional[float] = None) -> List[Tuple[float, float]]:
    logging.warning("ASTM D1160 to TBP conversion is a placeholder. Implement actual correlation.")
    # Convertir temperaturas de vacío a presión atmosférica y luego a TBP.
    # Esto NO es una conversión real.
    return d1160_data

def placeholder_convert_d2887_to_tbp(d2887_data: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    logging.warning("ASTM D2887 (SimDis) to TBP conversion is a placeholder. Implement actual correlation.")
    # SimDis es a menudo cercano a TBP, pero pueden ser necesarios ajustes.
    # Esto NO es una conversión real.
    return d2887_data

def placeholder_convert_d7169_to_tbp(d7169_data: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    logging.warning("ASTM D7169 (SimDis High Temp) to TBP conversion is a placeholder. Implement actual correlation.")
    # Similar a D2887 para crudos más pesados.
    # Esto NO es una conversión real.
    return d7169_data


def api_to_sg(api: Optional[float]) -> Optional[float]:
    """Convierte Gravedad API a Gravedad Específica (SG)."""
    if api is None:
        return None
    try:
        return 141.5 / (float(api) + 131.5)
    except (ValueError, TypeError):
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
        logging.warning(f"Could not convert SG value '{sg}' to float for API calculation.")
        return None

class CrudeOil:
    """
    Representa un crudo o una mezcla de crudos con sus propiedades básicas y curva de destilación.
    Ahora incluye manejo para diferentes tipos de curvas de destilación de entrada, convirtiéndolas a TBP.
    """
    def __init__(self, name: str, api_gravity: float, sulfur_content_wt_percent: float,
                 distillation_data_percent_vol_temp_C: List[Tuple[float, float]],
                 distillation_curve_type: str = "TBP", # Nuevo: TBP, ASTM D86, ASTM D1160, ASTM D2887, ASTM D7169
                 verbose: bool = False, is_blend: bool = False):
        self.name = name
        self.api = float(api_gravity)
        self.sg = api_to_sg(self.api)
        self.sulfur_total_wt_percent = float(sulfur_content_wt_percent)
        self.verbose = verbose
        self.is_blend = is_blend
        self.is_refinery_scenario_source = False # Mantener si se usa en otro lado
        self.original_feed_properties: Optional[Dict[str, Any]] = None # Mantener si se usa
        self.original_feed_components: Optional[List[Dict[str,Any]]] = None # Mantener si se usa

        self.original_distillation_data = distillation_data_percent_vol_temp_C
        self.original_distillation_curve_type = distillation_curve_type.upper() # Guardar en mayúsculas para consistencia

        # Convertir la curva de entrada a TBP para cálculos internos
        tbp_curve_data = self._convert_to_tbp(
            self.original_distillation_data,
            self.original_distillation_curve_type,
            self.api
        )

        if not tbp_curve_data: # Si la conversión falla o devuelve vacío
            logging.error(f"CrudeOil '{self.name}': TBP curve data is empty after conversion attempt from {self.original_distillation_curve_type}.")
            self.distillation_volumes_percent = np.array([])
            self.distillation_temperatures_C = np.array([])
            self.ibp_C = 0.0; self.fbp_C = 0.0; self.t50_C = 0.0; self.max_recovery_percent = 0.0
            return

        # El resto de la lógica de inicialización usa tbp_curve_data
        try:
            processed_dist_data = sorted(
                [(float(p[0]), float(p[1])) for p in tbp_curve_data if isinstance(p, (tuple, list)) and len(p) == 2],
                key=lambda x: x[0]
            )
        except (ValueError, TypeError) as e:
            logging.error(f"CrudeOil '{self.name}': Invalid format in TBP distillation_data after conversion. Error: {e}")
            processed_dist_data = [] # Asegurar que sea una lista vacía

        if not processed_dist_data:
            logging.error(f"CrudeOil '{self.name}': Distillation data is empty after processing TBP curve.")
            self.distillation_volumes_percent = np.array([])
            self.distillation_temperatures_C = np.array([])
            self.ibp_C = 0.0; self.fbp_C = 0.0; self.t50_C = 0.0; self.max_recovery_percent = 0.0
            return

        volumes = [p[0] for p in processed_dist_data]
        temps = [p[1] for p in processed_dist_data]

        # Lógica existente para asegurar IBP al 0% y FBP al 100%
        if 0.0 not in volumes:
            logging.warning(f"CrudeOil '{self.name}' (TBP): 0% vol (IBP) not found. Prepending with first point's temp @ 0%.")
            if temps: volumes.insert(0, 0.0); temps.insert(0, temps[0])
            else: volumes.insert(0, 0.0); temps.insert(0, 0.0) # Caso extremo: curva vacía

        if not volumes or volumes[-1] < 100.0: # Si está vacío o no llega al 100%
            if len(volumes) >= 2:
                delta_vol = volumes[-1] - volumes[-2]
                if delta_vol == 0: temp_at_100 = temps[-1]
                else: slope = (temps[-1] - temps[-2]) / delta_vol; temp_at_100 = temps[-1] + slope * (100.0 - volumes[-1])
                volumes.append(100.0); temps.append(temp_at_100)
            elif len(volumes) == 1: # Solo un punto (IBP)
                volumes.append(100.0); temps.append(temps[0]) # Asumir FBP = IBP
            else: # Curva vacía después de todo
                volumes.extend([0.0, 100.0]); temps.extend([0.0, 0.0])


        elif volumes[-1] > 100.0: # Si se pasa de 100%
            idx_over_100 = next((i for i, v in enumerate(volumes) if v > 100.0), len(volumes))
            volumes = volumes[:idx_over_100]; temps = temps[:idx_over_100]
            if not volumes or volumes[-1] < 100.0: # Si al cortar queda por debajo de 100
                 if len(volumes) >= 2:
                    delta_vol = volumes[-1] - volumes[-2]
                    if delta_vol == 0: temp_at_100 = temps[-1]
                    else: slope = (temps[-1] - temps[-2]) / delta_vol; temp_at_100 = temps[-1] + slope * (100.0 - volumes[-1])
                    volumes.append(100.0); temps.append(temp_at_100)
                 elif volumes: # Solo un punto después del corte
                     volumes.append(100.0); temps.append(temps[0])

        self.distillation_volumes_percent = np.array(volumes)
        self.distillation_temperatures_C = np.array(temps)

        if not self.distillation_volumes_percent.size:
            logging.error(f"CrudeOil '{self.name}': TBP Distillation curve empty after final processing.")
            self.ibp_C = 0.0; self.fbp_C = 0.0; self.t50_C = 0.0; self.max_recovery_percent = 0.0; return

        self.ibp_C = self.get_temperature_at_volume(0.0)
        self.fbp_C = self.get_temperature_at_volume(100.0)
        self.max_recovery_percent = 100.0 # Asumimos recuperación total para la curva TBP interna
        self.t50_C = self.get_temperature_at_volume(50.0)

        if self.t50_C <= 0 and self.verbose: logging.warning(f"CrudeOil '{self.name}': T50 ({self.t50_C}°C) <= 0 based on processed TBP.")
        if self.verbose:
            sg_display = f"{self.sg:.4f}" if self.sg is not None else "N/A"
            logging.info(f"CrudeOil '{self.name}' {'(Blend)' if self.is_blend else ''} initialized. API: {self.api:.1f}, SG: {sg_display}, Sulfur: {self.sulfur_total_wt_percent:.4f}% wt")
            logging.info(f"  Original Curve Type: {self.original_distillation_curve_type}, Processed as TBP.")
            logging.info(f"  TBP Dist curve ({len(self.distillation_volumes_percent)} pts): {self.ibp_C:.1f}°C (0%) to {self.fbp_C:.1f}°C (100%), T50: {self.t50_C:.1f}°C")

    def _convert_to_tbp(self, original_curve_data: List[Tuple[float, float]], curve_type: str, api_gravity: Optional[float]) -> List[Tuple[float, float]]:
        """
        Convierte la curva de destilación original a TBP.
        ESTA FUNCIÓN UTILIZA PLACEHOLDERS. DEBE IMPLEMENTAR CORRELACIONES REALES.
        """
        if not original_curve_data:
            logging.warning(f"Cannot convert empty original curve data for {self.name} (type: {curve_type}).")
            return []

        if curve_type == "TBP":
            return original_curve_data
        elif curve_type == "ASTM D86":
            return placeholder_convert_d86_to_tbp(original_curve_data, api_gravity)
        elif curve_type == "ASTM D1160":
            return placeholder_convert_d1160_to_tbp(original_curve_data, api_gravity)
        elif curve_type == "ASTM D2887":
            return placeholder_convert_d2887_to_tbp(original_curve_data)
        elif curve_type == "ASTM D7169":
            return placeholder_convert_d7169_to_tbp(original_curve_data)
        else:
            logging.warning(f"Unsupported distillation curve type '{curve_type}' for conversion. Using data as is (assumed TBP).")
            return original_curve_data

    def get_temperature_at_volume(self, percent_volume: float) -> float:
        if not self.distillation_volumes_percent.size: return 0.0
        # Asegurar que los puntos estén ordenados por volumen para np.interp
        sorted_indices = np.argsort(self.distillation_volumes_percent)
        sorted_vols = self.distillation_volumes_percent[sorted_indices]
        sorted_temps = self.distillation_temperatures_C[sorted_indices]

        left_val = sorted_temps[0] if sorted_temps.size > 0 else 0.0
        right_val = sorted_temps[-1] if sorted_temps.size > 0 else 0.0
        
        return float(np.interp(percent_volume, sorted_vols, sorted_temps, left=left_val, right=right_val))

    def get_volume_at_temperature(self, temperature_C: float) -> float:
        if not self.distillation_temperatures_C.size: return 0.0
        # Asegurar que los puntos estén ordenados por temperatura para np.interp
        # Si hay temperaturas duplicadas, np.interp puede comportarse de forma inesperada.
        # Una forma de manejarlo es eliminar duplicados manteniendo el primer volumen.
        unique_temps, unique_indices = np.unique(self.distillation_temperatures_C, return_index=True)
        unique_vols = self.distillation_volumes_percent[unique_indices]

        # Ahora ordenar por estas temperaturas únicas
        sorted_indices = np.argsort(unique_temps)
        sorted_temps = unique_temps[sorted_indices]
        sorted_vols = unique_vols[sorted_indices]

        left_val = sorted_vols[0] if sorted_vols.size > 0 else 0.0
        right_val = sorted_vols[-1] if sorted_vols.size > 0 else 100.0

        return float(np.interp(temperature_C, sorted_temps, sorted_vols, left=left_val, right=right_val))

    def __repr__(self):
        sg_display = f"{self.sg:.4f}" if self.sg is not None else "N/A"
        return (f"CrudeOil('{self.name}', API={self.api:.1f}, SG={sg_display}, S={self.sulfur_total_wt_percent}%, "
                f"OriginalCurve: {self.original_distillation_curve_type}, TBP_Pts={len(self.distillation_volumes_percent)})")

def create_blend_from_crudes(crude_components_data_list: List[Dict[str, Any]], verbose: bool = False) -> CrudeOil:
    if not crude_components_data_list: raise ValueError("Lista de componentes vacía.")
    total_prop = sum(float(c.get('proportion_vol', 0.0)) for c in crude_components_data_list)
    if not np.isclose(total_prop, 100.0): raise ValueError(f"Suma de proporciones ({total_prop}%) no es 100%.")

    components = []
    for comp_data in crude_components_data_list:
        dist_tuples = comp_data.get('distillation_data', [])
        dist_curve_type = comp_data.get('distillation_curve_type', "TBP") # Obtener el tipo de curva

        if not dist_tuples or not all(isinstance(p,(list,tuple)) and len(p)==2 and isinstance(p[0],(int,float)) and isinstance(p[1],(int,float)) for p in dist_tuples):
             raise ValueError(f"Formato/datos no numéricos en curva de destilación para {comp_data.get('name', 'Desconocido')}")
        try:
            crude_obj = CrudeOil(
                name=str(comp_data.get('name','?')),
                api_gravity=float(comp_data.get('api',0.0)),
                sulfur_content_wt_percent=float(comp_data.get('sulfur',0.0)),
                distillation_data_percent_vol_temp_C=[(float(p[0]), float(p[1])) for p in dist_tuples],
                distillation_curve_type=dist_curve_type, # Pasar el tipo de curva
                verbose=verbose
            )
        except Exception as e: raise ValueError(f"Error creando CrudeOil para {comp_data.get('name','?')}: {e}")

        if crude_obj.sg is None: raise ValueError(f"SG no calculado para {crude_obj.name} (API: {crude_obj.api}).")
        components.append({'obj': crude_obj, 'proportion_vol': float(comp_data.get('proportion_vol',0.0))/100.0})

    # Los cálculos de propiedades de la mezcla se basan en las propiedades individuales
    # y las curvas TBP (ya convertidas dentro de cada CrudeOil)
    sg_b = sum(c['obj'].sg * c['proportion_vol'] for c in components if c['obj'].sg is not None)
    api_b = sg_to_api(sg_b)
    if api_b is None: raise ValueError("API de mezcla no calculable.")

    sulfur_b = sum(c['obj'].sulfur_total_wt_percent * (c['proportion_vol'] * c['obj'].sg / sg_b) for c in components if sg_b!=0 and c['obj'].sg is not None)

    # La curva TBP de la mezcla se calcula promediando las temperaturas de las curvas TBP de los componentes
    vols_blend = np.linspace(0,100,21) # Puntos estándar para la curva de mezcla
    temps_blend = []
    for v_target in vols_blend:
        weighted_temp_sum = 0
        for c in components:
            # c['obj'].get_temperature_at_volume() opera sobre la curva TBP interna del componente
            weighted_temp_sum += c['obj'].get_temperature_at_volume(v_target) * c['proportion_vol']
        temps_blend.append(weighted_temp_sum)

    dist_b_tbp = list(zip(vols_blend, temps_blend))
    name_b = "Mezcla ("+", ".join([f"{c['obj'].name} {c['proportion_vol']*100:.0f}%" for c in components])+")"

    if verbose: logging.info(f"Blend '{name_b}': API {api_b:.1f}, SG {sg_b:.4f}, S {sulfur_b:.4f}%wt. Blend TBP curve calculated.")

    # La mezcla se crea directamente con su curva TBP calculada.
    # El 'distillation_curve_type' de la mezcla será "TBP".
    return CrudeOil(name_b, api_b, sulfur_b, dist_b_tbp, distillation_curve_type="TBP", verbose=verbose, is_blend=True)


class DistillationCut:
    def __init__(self, name: str, t_initial_C: float, t_final_C: float, crude_oil_ref: CrudeOil):
        self.name = name; self.t_initial_C = float(t_initial_C); self.t_final_C = float(t_final_C)
        self.crude_oil = crude_oil_ref; self.yield_vol_percent: Optional[float]=0.0; self.vabp_C: Optional[float]=None
        self.api_cut: Optional[float]=None; self.sg_cut: Optional[float]=None; self.yield_wt_percent: Optional[float]=0.0
        self.sulfur_cut_wt_percent: Optional[float]=0.0; self.sulfur_cut_ppm: Optional[float]=0.0
        self.distillation_curve: Optional[List[Tuple[float,float]]]=None
        # La propiedad is_gas_cut se basa en la temperatura final del corte, no en el tipo de curva del crudo
        self.is_gas_cut = name.lower().startswith("gas") and self.t_final_C<=40 # Ajustar si es necesario

    def _estimate_distillation_curve(self):
        if self.yield_vol_percent is None or self.yield_vol_percent<=1e-6 or self.t_final_C<=self.t_initial_C: return None
        vabp = self.vabp_C if self.vabp_C is not None else (self.t_initial_C+self.t_final_C)/2.0
        if self.is_gas_cut: return [(0.0,self.t_initial_C),(50.0,vabp),(100.0,self.t_final_C)] # Curva simple para gases
        # Estimación de curva para productos líquidos (podría mejorarse con correlaciones)
        vols=[0,5,10,30,50,70,90,95,100.0]; temps=[]
        for v in vols:
            # Función sigmoide simple para dar forma a la curva entre T_inicial, VABP y T_final
            # Esto es una simplificación y podría reemplazarse por métodos más rigurosos.
            if v < 50: # De IBP a VABP
                f = v / 50.0 # Fracción lineal hasta VABP
                t = self.t_initial_C + f * (vabp - self.t_initial_C)
            else: # De VABP a FBP
                f = (v - 50.0) / 50.0 # Fracción lineal desde VABP
                t = vabp + f * (self.t_final_C - vabp)
            temps.append(float(t))
        return list(zip(vols,temps))

    def _estimate_api_cut_placeholder(self, api_sensitivity_factor: float = 7.0) -> Optional[float]:
        if self.is_gas_cut: return 110.0 # API típico para gases (muy ligero)
        if self.vabp_C is None: return None
        # crude_oil.t50_C y crude_oil.api se refieren a la curva TBP del crudo
        t50_crude_tbp = self.crude_oil.t50_C if self.crude_oil.t50_C > 0 else self.vabp_C # Usar VABP del corte si T50 del crudo no es válido
        tdiff_factor = (self.vabp_C - t50_crude_tbp) / 100.0 # Factor de diferencia de temperatura
        api_est = self.crude_oil.api - (tdiff_factor * api_sensitivity_factor)
        return max(0.1,min(api_est,120.0)) # Asegurar que esté en un rango razonable

    def calculate_basic_properties(self, api_sensitivity_factor: float = 7.0):
        if self.is_gas_cut and self.t_initial_C > -50: self.t_initial_C=min(self.t_initial_C,-40.0) # Ajustar T_inicial para gases

        # Los volúmenes se obtienen de la curva TBP del crudo
        v_init=self.crude_oil.get_volume_at_temperature(self.t_initial_C)
        v_fin=self.crude_oil.get_volume_at_temperature(self.t_final_C)
        self.yield_vol_percent=abs(v_fin-v_init)

        if self.yield_vol_percent is not None and self.yield_vol_percent > 1e-6:
            if self.is_gas_cut: self.vabp_C=(self.t_initial_C+self.t_final_C)/2.0
            else: # El VABP se calcula sobre la curva TBP del crudo
                self.vabp_C=self.crude_oil.get_temperature_at_volume(v_init+(self.yield_vol_percent/2.0))
        else: self.vabp_C=(self.t_initial_C+self.t_final_C)/2.0

        if self.vabp_C is not None or self.is_gas_cut:
            self.api_cut=self._estimate_api_cut_placeholder(api_sensitivity_factor=api_sensitivity_factor)
        self.sg_cut=api_to_sg(self.api_cut)

        if self.yield_vol_percent is not None and self.sg_cut is not None and self.crude_oil.sg is not None and self.crude_oil.sg>1e-6:
            # Factor de corrección para gases (menor densidad)
            density_correction_factor = 0.85 if self.is_gas_cut else 1.0
            self.yield_wt_percent = self.yield_vol_percent * (self.sg_cut / self.crude_oil.sg) * density_correction_factor
        else: self.yield_wt_percent=0.0

        self.distillation_curve=self._estimate_distillation_curve() # Estimar curva TBP del corte

    def set_sulfur_properties(self, s_wt:Optional[float]):
        if s_wt is not None:
            # Ajuste simple para azufre en gases (menor concentración)
            s_effective = min(float(s_wt) * 0.3, 0.1) if self.is_gas_cut else float(s_wt)
            self.sulfur_cut_wt_percent = s_effective
            self.sulfur_cut_ppm = s_effective * 10000.0
        else:
            self.sulfur_cut_wt_percent=0.0; self.sulfur_cut_ppm=0.0

    def get_distillation_data(self): return pd.DataFrame(self.distillation_curve,columns=["Volumen (%)","Temperatura (°C)"]) if self.distillation_curve else pd.DataFrame(columns=["Volumen (%)","Temperatura (°C)"])
    def to_dict(self): return {"Corte":self.name,"T Inicial (°C)":self.t_initial_C,"T Final (°C)":self.t_final_C,"Rend. Vol (%)":self.yield_vol_percent,
                               "Rend. Peso (%)":self.yield_wt_percent,"VABP (°C)":self.vabp_C,"API Corte":self.api_cut,"SG Corte":self.sg_cut,
                               "Azufre (%peso)":self.sulfur_cut_wt_percent,"Azufre (ppm)":self.sulfur_cut_ppm}
    def __repr__(self): return f"DistillationCut('{self.name}',YVol={self.yield_vol_percent:.2f if self.yield_vol_percent is not None else 'NA'}%,S_wt={self.sulfur_cut_wt_percent:.4f if self.sulfur_cut_wt_percent is not None else 'NA'}%)"

# --- Funciones de Cálculo de Torres ---
# Estas funciones operan sobre un CrudeOil que ya tiene su curva procesada como TBP.
# No necesitan conocer el tipo de curva original del crudo.

def calculate_distillation_cuts(crude_oil: CrudeOil, cut_definitions: List[Tuple[str, float]],
                                verbose: bool = False, api_sensitivity_factor: float = 7.0,
                                empirical_data: Optional[Dict[str, Any]] = None) -> List[DistillationCut]:
    """
    Calcula los cortes de destilación para un crudo dado (que ya tiene su curva como TBP).
    """
    if verbose: logging.info(f"\n--- Calculating Cuts for Feed '{crude_oil.name}' (API Sens: {api_sensitivity_factor}) ---")
    # crude_oil.ibp_C, crude_oil.fbp_C, crude_oil.t50_C ya están basados en la TBP interna.

    if empirical_data and isinstance(empirical_data.get('distribution_data'), dict) and empirical_data['distribution_data'].get('cuts'):
        logging.info(f"Applying empirical cut distribution for {crude_oil.name} based on provided empirical_data.")
        return apply_empirical_distribution(crude_oil, cut_definitions, empirical_data, verbose, api_sensitivity_factor)

    cuts_output = []
    current_t_initial_C = crude_oil.ibp_C # IBP de la TBP del crudo
    for i, (cut_name, cut_t_final_C_val) in enumerate(cut_definitions):
        t_start = current_t_initial_C; t_end = float(cut_t_final_C_val)
        cut = DistillationCut(name=cut_name, t_initial_C=t_start, t_final_C=t_end, crude_oil_ref=crude_oil)
        cut.calculate_basic_properties(api_sensitivity_factor=api_sensitivity_factor)
        if t_end < t_start:
            if verbose: logging.warning(f"T_final ({t_end}°C) for '{cut_name}' < T_initial ({t_start}°C). Yield set to 0.")
            cut.yield_vol_percent = 0.0; cut.yield_wt_percent = 0.0
        cuts_output.append(cut)
        current_t_initial_C = t_end

    if current_t_initial_C < crude_oil.fbp_C : # FBP de la TBP del crudo
        residue_name_inferred = f"Residuo de {crude_oil.name}"
        if not (cut_definitions and cut_definitions[-1][0].lower().startswith("residuo")):
            residue_cut = DistillationCut(name=residue_name_inferred, t_initial_C=current_t_initial_C, t_final_C=crude_oil.fbp_C, crude_oil_ref=crude_oil)
            residue_cut.calculate_basic_properties(api_sensitivity_factor=api_sensitivity_factor)
            vol_at_residue_start = crude_oil.get_volume_at_temperature(current_t_initial_C)
            residue_cut.yield_vol_percent = 100.0 - vol_at_residue_start # Rendimiento sobre base TBP 100%
            if residue_cut.sg_cut and crude_oil.sg and crude_oil.sg > 1e-6 and residue_cut.yield_vol_percent is not None:
                 residue_cut.yield_wt_percent = residue_cut.yield_vol_percent * (residue_cut.sg_cut / crude_oil.sg)
            else: residue_cut.yield_wt_percent = 0.0
            cuts_output.append(residue_cut)

    total_crude_sulfur = crude_oil.sulfur_total_wt_percent
    if total_crude_sulfur > 0:
        total_yield_wt = sum(c.yield_wt_percent for c in cuts_output if c.yield_wt_percent is not None)
        if total_yield_wt > 1e-6:
            denominator = sum(((c.vabp_C / crude_oil.t50_C if crude_oil.t50_C > 1e-6 and c.vabp_C is not None else 1.0) * (c.yield_wt_percent if c.yield_wt_percent is not None else 0.0))
                              for c in cuts_output if c.yield_wt_percent is not None and c.yield_wt_percent > 0)
            if abs(denominator) > 1e-9:
                factor_s = (total_crude_sulfur * total_yield_wt) / denominator
                for cut in cuts_output:
                    if cut.vabp_C is not None and cut.yield_wt_percent is not None and cut.yield_wt_percent > 0:
                        vabp_ratio = cut.vabp_C / crude_oil.t50_C if crude_oil.t50_C > 1e-6 else 1.0
                        sulfur_c = factor_s * vabp_ratio
                        cut.set_sulfur_properties(max(0, min(sulfur_c, total_crude_sulfur * 10)))
                    else: cut.set_sulfur_properties(0.0)
            elif verbose:
                logging.warning(f"Sulfur model failed for {crude_oil.name}. Distributing S uniformly by weight yield.")
                for cut in cuts_output:
                    s_dist = total_crude_sulfur * (cut.yield_wt_percent / total_yield_wt) if cut.yield_wt_percent and cut.yield_wt_percent > 0 and total_yield_wt > 0 else 0.0
                    cut.set_sulfur_properties(s_dist)
        else:
            for cut in cuts_output: cut.set_sulfur_properties(0.0)
    else:
        for cut in cuts_output: cut.set_sulfur_properties(0.0)
    return cuts_output


def calculate_atmospheric_cuts(crude_oil_feed: CrudeOil,
                               atmospheric_cut_definitions: List[Tuple[str, float]],
                               verbose: bool = False,
                               api_sensitivity_factor: float = 7.0,
                               empirical_data_for_crude: Optional[Dict[str, Any]] = None
                               ) -> Tuple[List[DistillationCut], Optional[DistillationCut]]:
    if verbose: logging.info(f"\n--- Calculating Atmospheric Cuts for '{crude_oil_feed.name}' ---")
    # crude_oil_feed ya tiene su curva como TBP internamente
    all_atmospheric_products = calculate_distillation_cuts(
        crude_oil=crude_oil_feed,
        cut_definitions=atmospheric_cut_definitions,
        verbose=verbose,
        api_sensitivity_factor=api_sensitivity_factor,
        empirical_data=empirical_data_for_crude
    )
    # Lógica para separar destilados y residuo (sin cambios necesarios aquí)
    atmospheric_distillates = []
    atmospheric_residue_object = None
    if all_atmospheric_products:
        if atmospheric_cut_definitions and atmospheric_cut_definitions[-1][0].lower().startswith("residuo atmosf"):
            found_residue = False
            temp_distillates = []
            for prod in all_atmospheric_products:
                if prod.name.lower().startswith("residuo atmosf"):
                    atmospheric_residue_object = prod
                    found_residue = True
                else:
                    temp_distillates.append(prod)
            atmospheric_distillates = temp_distillates
            if not found_residue and all_atmospheric_products:
                 atmospheric_residue_object = all_atmospheric_products[-1]
                 atmospheric_distillates = all_atmospheric_products[:-1]
        elif all_atmospheric_products[-1].name.startswith("Residuo de"):
            atmospheric_residue_object = all_atmospheric_products[-1]
            atmospheric_distillates = all_atmospheric_products[:-1]
        else:
            if len(all_atmospheric_products) == len(atmospheric_cut_definitions):
                 atmospheric_distillates = all_atmospheric_products
            elif len(all_atmospheric_products) > len(atmospheric_cut_definitions):
                 atmospheric_residue_object = all_atmospheric_products[-1]
                 atmospheric_distillates = all_atmospheric_products[:-1]
            else:
                 atmospheric_distillates = all_atmospheric_products
    return atmospheric_distillates, atmospheric_residue_object


def create_vacuum_feed_from_residue(original_crude: CrudeOil, # CrudeOil original (con su TBP)
                                    atmospheric_residue_cut: Optional[DistillationCut],
                                    verbose: bool = False) -> Optional[CrudeOil]:
    if atmospheric_residue_cut is None or atmospheric_residue_cut.yield_vol_percent is None or atmospheric_residue_cut.yield_vol_percent <= 1e-6:
        if verbose: logging.info("Residuo atmosférico nulo o con rendimiento cero. No se crea alimentación a vacío.")
        return None
    if atmospheric_residue_cut.api_cut is None or atmospheric_residue_cut.sulfur_cut_wt_percent is None:
        logging.error(f"No se puede crear alim. vacío desde '{atmospheric_residue_cut.name}', faltan API o Azufre."); return None

    # Las temperaturas del residuo atmosférico (t_initial_C, t_final_C) se refieren a la escala TBP del crudo original.
    t_start_res_abs_tbp = atmospheric_residue_cut.t_initial_C # Esta es TBP equivalente
    t_end_res_abs_tbp = original_crude.fbp_C # El FBP del crudo original (TBP)

    # Volúmenes en la curva TBP del crudo original
    vol_start_res_orig_tbp = original_crude.get_volume_at_temperature(t_start_res_abs_tbp)
    total_vol_res_orig_scale_tbp = 100.0 - vol_start_res_orig_tbp

    if total_vol_res_orig_scale_tbp <= 1e-6:
        if verbose: logging.info(f"Volumen de residuo atm. en escala TBP ({total_vol_res_orig_scale_tbp}%) muy pequeño. No se crea alim. vacío."); return None

    vac_feed_tbp_curve = []
    for new_vol_pct_on_vac_feed in np.linspace(0, 100, 21): # 0-100% de la alimentación a vacío
        # Convertir %vol en alim. vacío a %vol en crudo original (escala TBP)
        orig_equiv_vol_tbp = vol_start_res_orig_tbp + (new_vol_pct_on_vac_feed / 100.0) * total_vol_res_orig_scale_tbp
        orig_equiv_vol_tbp = min(orig_equiv_vol_tbp, 100.0) # No exceder 100% del crudo original

        # Obtener la temperatura TBP correspondiente del crudo original
        temp_at_new_vol_tbp = original_crude.get_temperature_at_volume(orig_equiv_vol_tbp)
        vac_feed_tbp_curve.append((new_vol_pct_on_vac_feed, temp_at_new_vol_tbp))

    if vac_feed_tbp_curve: # Asegurar que IBP y FBP de la nueva curva sean consistentes
        vac_feed_tbp_curve[0] = (0.0, t_start_res_abs_tbp) # IBP de la alim. vacío es T inicial del residuo (TBP)
        vac_feed_tbp_curve[-1] = (100.0, t_end_res_abs_tbp) # FBP de la alim. vacío es FBP del crudo original (TBP)

    vac_feed_name = f"Alim. Vacío de {original_crude.name}"
    if verbose: logging.info(f"Creando alim. vacío: {vac_feed_name}. API: {atmospheric_residue_cut.api_cut:.1f}, S: {atmospheric_residue_cut.sulfur_cut_wt_percent:.4f}%")

    # La alimentación a vacío se crea con su propia curva TBP.
    # El tipo de curva es "TBP" por definición.
    return CrudeOil(name=vac_feed_name, api_gravity=atmospheric_residue_cut.api_cut,
                    sulfur_content_wt_percent=atmospheric_residue_cut.sulfur_cut_wt_percent,
                    distillation_data_percent_vol_temp_C=vac_feed_tbp_curve,
                    distillation_curve_type="TBP", # Es una curva TBP por construcción
                    verbose=verbose, is_blend=original_crude.is_blend)


def calculate_vacuum_cuts(vacuum_feed: CrudeOil, # vacuum_feed ya es un CrudeOil con curva TBP
                          vacuum_cut_definitions: List[Tuple[str, float]],
                          verbose: bool = False,
                          api_sensitivity_factor: float = 7.0
                          ) -> List[DistillationCut]:
    if verbose: logging.info(f"\n--- Calculating Vacuum Cuts for '{vacuum_feed.name}' ---")
    # vacuum_feed.ibp_C, etc., ya son TBP.
    vacuum_products = calculate_distillation_cuts(
        crude_oil=vacuum_feed, cut_definitions=vacuum_cut_definitions,
        verbose=verbose, api_sensitivity_factor=api_sensitivity_factor,
        empirical_data=None # No se suelen aplicar datos empíricos de crudo aquí
    )
    return vacuum_products


def apply_empirical_distribution(crude_oil: CrudeOil, cut_definitions: List[Tuple[str, float]],
                                 empirical_data: Dict[str, Any], verbose: bool = False,
                                 api_sensitivity_factor: float = 7.0) -> List[DistillationCut]:
    if verbose: logging.info(f"Applying empirical distribution for {crude_oil.name}")
    cuts_output = []
    empirical_cuts_list = empirical_data.get('distribution_data', {}).get('cuts', [])
    empirical_cuts_map = {cut_data.get("Corte"): cut_data for cut_data in empirical_cuts_list if isinstance(cut_data, dict) and cut_data.get("Corte")}

    def _safe_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
        if value is None or (isinstance(value, str) and value.strip().upper() in ["N/A", "NA", ""]): return default
        try: return float(value)
        except (ValueError, TypeError):
            logging.warning(f"Empirical apply: Could not convert '{value}' to float, using default {default}.")
            return default

    current_t_initial_C = crude_oil.ibp_C # IBP de la TBP del crudo
    total_empirical_yield_vol = 0.0
    processed_empirical_cuts = set()

    for cut_name, cut_t_final_C_val in cut_definitions:
        t_start = current_t_initial_C; t_end = float(cut_t_final_C_val)
        cut = DistillationCut(name=cut_name, t_initial_C=t_start, t_final_C=t_end, crude_oil_ref=crude_oil)

        if cut_name in empirical_cuts_map:
            emp_cut = empirical_cuts_map[cut_name]
            if verbose: logging.info(f"  Using empirical data for cut '{cut_name}'")
            cut.yield_vol_percent = _safe_float(emp_cut.get("Rend. Vol (%)"))
            cut.yield_wt_percent = _safe_float(emp_cut.get("Rend. Peso (%)"))
            cut.api_cut = _safe_float(emp_cut.get("API Corte"), None)
            cut.sg_cut = api_to_sg(cut.api_cut)
            cut.vabp_C = _safe_float(emp_cut.get("VABP (°C)"), (t_start + t_end) / 2.0)
            cut.set_sulfur_properties(_safe_float(emp_cut.get("Azufre (%peso)")))
            cut.distillation_curve = cut._estimate_distillation_curve() # Estimar curva TBP del corte
            total_empirical_yield_vol += cut.yield_vol_percent if cut.yield_vol_percent is not None else 0.0
            processed_empirical_cuts.add(cut_name)
        else:
            if verbose: logging.info(f"  No empirical data for cut '{cut_name}', calculating theoretically.")
            cut.calculate_basic_properties(api_sensitivity_factor=api_sensitivity_factor)
        cuts_output.append(cut)
        current_t_initial_C = t_end

    if current_t_initial_C < crude_oil.fbp_C: # FBP de la TBP del crudo
        residue_name_inferred = f"Residuo de {crude_oil.name}"
        is_residue_defined = (cut_definitions and cut_definitions[-1][0].lower().startswith("residuo")) or \
                             any(name.lower().startswith("residuo") for name in empirical_cuts_map if name not in processed_empirical_cuts)
        if not is_residue_defined:
             residue_cut = DistillationCut(name=residue_name_inferred, t_initial_C=current_t_initial_C, t_final_C=crude_oil.fbp_C, crude_oil_ref=crude_oil)
             residue_cut.calculate_basic_properties(api_sensitivity_factor=api_sensitivity_factor)
             vol_at_residue_start = crude_oil.get_volume_at_temperature(current_t_initial_C)
             residue_cut.yield_vol_percent = 100.0 - vol_at_residue_start
             if residue_cut.sg_cut and crude_oil.sg and crude_oil.sg > 1e-6 and residue_cut.yield_vol_percent is not None:
                 residue_cut.yield_wt_percent = residue_cut.yield_vol_percent * (residue_cut.sg_cut / crude_oil.sg)
             else: residue_cut.yield_wt_percent = 0.0
             cuts_output.append(residue_cut)
             if verbose: logging.info(f"  Calculated theoretical residue '{residue_cut.name}'")

    unused_empirical_cuts = set(empirical_cuts_map.keys()) - processed_empirical_cuts
    if unused_empirical_cuts:
        logging.warning(f"Empirical cuts defined but not matched to theoretical definitions: {', '.join(unused_empirical_cuts)}")
    return cuts_output


# --- Código de Prueba (sin cambios significativos, pero ahora CrudeOil toma el tipo de curva) ---
if __name__ == "__main__":
    # Ejemplo con una curva que podría ser D86 (requeriría conversión real)
    crude1_data_points_d86_example = [(0, 35), (10, 100), (30, 200), (50, 280), (70, 360), (90, 450), (95, 500)] # Simula D86 (más bajo IBP)
    test_crude_obj_d86 = CrudeOil(name="Crudo Ejemplo (Simula D86)", api_gravity=38.0, # API más alto para D86
                              sulfur_content_wt_percent=0.05,
                              distillation_data_percent_vol_temp_C=crude1_data_points_d86_example,
                              distillation_curve_type="ASTM D86", # Especificar tipo
                              verbose=True)

    # Ejemplo con una curva TBP directa
    crude2_data_points_tbp = [(0, 50), (10, 120), (30, 220), (50, 300), (70, 380), (90, 480), (100, 550)]
    test_crude_obj_tbp = CrudeOil(name="Crudo Ejemplo (TBP Directa)", api_gravity=35.0,
                              sulfur_content_wt_percent=0.10,
                              distillation_data_percent_vol_temp_C=crude2_data_points_tbp,
                              distillation_curve_type="TBP", # Especificar tipo
                              verbose=True)

    atm_cut_defs = [("Nafta Ligera", 90.0), ("Nafta Pesada", 175.0), ("Kerosene", 230.0), ("Gasóleo Atmosférico", 350.0)]

    logging.info("\n--- Prueba Torre Atmosférica con Crudo (Simula D86) ---")
    atm_distillates_d86, atm_residue_d86 = calculate_atmospheric_cuts(test_crude_obj_d86, atm_cut_defs, verbose=True, api_sensitivity_factor=7.0)
    for c in atm_distillates_d86: logging.info(f"  Distillate: {c.name}: Rend Vol {c.yield_vol_percent:.2f}%, API {c.api_cut:.1f if c.api_cut else 'N/A'}")
    if atm_residue_d86: logging.info(f"  Residue: {atm_residue_d86.name}: Rend Vol {atm_residue_d86.yield_vol_percent:.2f}%, API {atm_residue_d86.api_cut:.1f if atm_residue_d86.api_cut else 'N/A'}")


    logging.info("\n--- Prueba Torre Atmosférica con Crudo (TBP Directa) ---")
    atm_distillates_tbp, atm_residue_tbp = calculate_atmospheric_cuts(test_crude_obj_tbp, atm_cut_defs, verbose=True, api_sensitivity_factor=7.0)
    for c in atm_distillates_tbp: logging.info(f"  Distillate: {c.name}: Rend Vol {c.yield_vol_percent:.2f}%, API {c.api_cut:.1f if c.api_cut else 'N/A'}")
    if atm_residue_tbp:
        logging.info(f"  Residue: {atm_residue_tbp.name}: Rend Vol {atm_residue_tbp.yield_vol_percent:.2f}%, API {atm_residue_tbp.api_cut:.1f if atm_residue_tbp.api_cut else 'N/A'}")
        logging.info("\n--- Creando Alimentación a Vacío (desde TBP) ---")
        vacuum_feed_tbp = create_vacuum_feed_from_residue(test_crude_obj_tbp, atm_residue_tbp, verbose=True)
        if vacuum_feed_tbp:
            vac_cut_defs = [("VGO Ligero", 450.0), ("VGO Pesado", 550.0)] # Temperaturas TBP equivalentes
            logging.info("\n--- Prueba Torre de Vacío (desde TBP) ---")
            vacuum_products_tbp = calculate_vacuum_cuts(vacuum_feed_tbp, vac_cut_defs, verbose=True, api_sensitivity_factor=7.0)
            for vp in vacuum_products_tbp:
                 yield_on_original_crude = (vp.yield_vol_percent / 100.0) * (atm_residue_tbp.yield_vol_percent / 100.0) * 100.0 if vp.yield_vol_percent and atm_residue_tbp.yield_vol_percent else 0.0
                 logging.info(f"  Vac Product: {vp.name}: Rend Vol (on VacFeed) {vp.yield_vol_percent:.2f}%, Rend Vol (on OrigCrude) {yield_on_original_crude:.2f}%, API {vp.api_cut:.1f if vp.api_cut else 'N/A'}")
    else:
        logging.info("No se generó residuo atmosférico para procesar en vacío (desde TBP).")

