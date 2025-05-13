import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    """
    def __init__(self, name: str, api_gravity: float, sulfur_content_wt_percent: float,
                 distillation_data_percent_vol_temp_C: List[Tuple[float, float]],
                 verbose: bool = False, is_blend: bool = False):
        self.name = name
        self.api = float(api_gravity)
        self.sg = api_to_sg(self.api)
        self.sulfur_total_wt_percent = float(sulfur_content_wt_percent)
        self.verbose = verbose
        self.is_blend = is_blend
        self.is_refinery_scenario_source = False
        self.original_feed_properties: Optional[Dict[str, Any]] = None
        self.original_feed_components: Optional[List[Dict[str,Any]]] = None

        if not distillation_data_percent_vol_temp_C:
            logging.error(f"CrudeOil '{self.name}': Distillation data cannot be empty.")
            self.distillation_volumes_percent = np.array([])
            self.distillation_temperatures_C = np.array([])
            self.ibp_C = 0.0; self.fbp_C = 0.0; self.t50_C = 0.0; self.max_recovery_percent = 0.0
            return
        try:
            processed_dist_data = sorted(
                [(float(p[0]), float(p[1])) for p in distillation_data_percent_vol_temp_C if isinstance(p, (tuple, list)) and len(p) == 2],
                key=lambda x: x[0]
            )
        except (ValueError, TypeError) as e:
            logging.error(f"CrudeOil '{self.name}': Invalid format in distillation_data. Error: {e}")
            self.distillation_volumes_percent = np.array([])
            self.distillation_temperatures_C = np.array([])
            self.ibp_C = 0.0; self.fbp_C = 0.0; self.t50_C = 0.0; self.max_recovery_percent = 0.0
            return
        if not processed_dist_data:
            logging.error(f"CrudeOil '{self.name}': Distillation data is empty after processing.")
            self.distillation_volumes_percent = np.array([])
            self.distillation_temperatures_C = np.array([])
            self.ibp_C = 0.0; self.fbp_C = 0.0; self.t50_C = 0.0; self.max_recovery_percent = 0.0
            return
        volumes = [p[0] for p in processed_dist_data]
        temps = [p[1] for p in processed_dist_data]
        if 0.0 not in volumes:
            logging.warning(f"CrudeOil '{self.name}': 0% vol (IBP) not found. Prepending with first point's temp @ 0%.")
            volumes.insert(0, 0.0); temps.insert(0, temps[0])
        if volumes[-1] < 100.0:
            if len(volumes) >= 2:
                delta_vol = volumes[-1] - volumes[-2]
                if delta_vol == 0: temp_at_100 = temps[-1]
                else: slope = (temps[-1] - temps[-2]) / delta_vol; temp_at_100 = temps[-1] + slope * (100.0 - volumes[-1])
                volumes.append(100.0); temps.append(temp_at_100)
            elif len(volumes) == 1: volumes.append(100.0); temps.append(temps[0])
        elif volumes[-1] > 100.0:
            idx_over_100 = next((i for i, v in enumerate(volumes) if v > 100.0), len(volumes))
            volumes = volumes[:idx_over_100]; temps = temps[:idx_over_100]
            if not volumes or volumes[-1] < 100.0:
                 if len(volumes) >= 2:
                    delta_vol = volumes[-1] - volumes[-2]
                    if delta_vol == 0: temp_at_100 = temps[-1]
                    else: slope = (temps[-1] - temps[-2]) / delta_vol; temp_at_100 = temps[-1] + slope * (100.0 - volumes[-1])
                    volumes.append(100.0); temps.append(temp_at_100)
                 elif volumes: volumes.append(100.0); temps.append(temps[0])
        self.distillation_volumes_percent = np.array(volumes)
        self.distillation_temperatures_C = np.array(temps)
        if not self.distillation_volumes_percent.size:
            logging.error(f"CrudeOil '{self.name}': Distillation curve empty after final processing.")
            self.ibp_C = 0.0; self.fbp_C = 0.0; self.t50_C = 0.0; self.max_recovery_percent = 0.0; return
        self.ibp_C = self.get_temperature_at_volume(0.0); self.fbp_C = self.get_temperature_at_volume(100.0)
        self.max_recovery_percent = 100.0; self.t50_C = self.get_temperature_at_volume(50.0)
        if self.t50_C <= 0 and self.verbose: logging.warning(f"CrudeOil '{self.name}': T50 ({self.t50_C}°C) <= 0.")
        if self.verbose:
            sg_display = f"{self.sg:.4f}" if self.sg is not None else "N/A"
            logging.info(f"CrudeOil '{self.name}' {'(Blend)' if self.is_blend else ''} initialized. API: {self.api:.1f}, SG: {sg_display}, Sulfur: {self.sulfur_total_wt_percent:.4f}% wt")
            logging.info(f"  Dist curve ({len(self.distillation_volumes_percent)} pts): {self.ibp_C:.1f}°C (0%) to {self.fbp_C:.1f}°C (100%), T50: {self.t50_C:.1f}°C")
    def get_temperature_at_volume(self, percent_volume: float) -> float:
        if not self.distillation_volumes_percent.size: return 0.0
        left = self.distillation_temperatures_C[0] if self.distillation_temperatures_C.size > 0 else 0.0
        right = self.distillation_temperatures_C[-1] if self.distillation_temperatures_C.size > 0 else 0.0
        return float(np.interp(percent_volume, self.distillation_volumes_percent, self.distillation_temperatures_C, left=left, right=right))
    def get_volume_at_temperature(self, temperature_C: float) -> float:
        if not self.distillation_temperatures_C.size: return 0.0
        left = self.distillation_volumes_percent[0] if self.distillation_volumes_percent.size > 0 else 0.0
        right = self.distillation_volumes_percent[-1] if self.distillation_volumes_percent.size > 0 else 100.0
        return float(np.interp(temperature_C, self.distillation_temperatures_C, self.distillation_volumes_percent, left=left, right=right))
    def __repr__(self):
        sg_display = f"{self.sg:.4f}" if self.sg is not None else "N/A"
        return (f"CrudeOil('{self.name}', API={self.api:.1f}, SG={sg_display}, S={self.sulfur_total_wt_percent}%, Pts={len(self.distillation_volumes_percent)})")

def create_blend_from_crudes(crude_components_data_list: List[Dict[str, Any]], verbose: bool = False) -> CrudeOil:
    if not crude_components_data_list: raise ValueError("Lista de componentes vacía.")
    total_prop = sum(float(c.get('proportion_vol', 0.0)) for c in crude_components_data_list)
    if not np.isclose(total_prop, 100.0): raise ValueError(f"Suma de proporciones ({total_prop}%) no es 100%.")
    components = []
    for comp_data in crude_components_data_list:
        dist_tuples = comp_data.get('distillation_data', [])
        if not dist_tuples or not all(isinstance(p,(list,tuple)) and len(p)==2 and isinstance(p[0],(int,float)) and isinstance(p[1],(int,float)) for p in dist_tuples):
             raise ValueError(f"Formato/datos no numéricos en curva TBP para {comp_data.get('name', 'Desconocido')}")
        try:
            crude_obj = CrudeOil(str(comp_data.get('name','?')), float(comp_data.get('api',0.0)), float(comp_data.get('sulfur',0.0)),
                                 [(float(p[0]), float(p[1])) for p in dist_tuples], verbose)
        except Exception as e: raise ValueError(f"Error creando CrudeOil para {comp_data.get('name','?')}: {e}")
        if crude_obj.sg is None: raise ValueError(f"SG no calculado para {crude_obj.name} (API: {crude_obj.api}).")
        components.append({'obj': crude_obj, 'proportion_vol': float(comp_data.get('proportion_vol',0.0))/100.0})
    sg_b = sum(c['obj'].sg * c['proportion_vol'] for c in components if c['obj'].sg is not None)
    api_b = sg_to_api(sg_b);
    if api_b is None: raise ValueError("API de mezcla no calculable.")
    sulfur_b = sum(c['obj'].sulfur_total_wt_percent * (c['proportion_vol'] * c['obj'].sg / sg_b) for c in components if sg_b!=0 and c['obj'].sg is not None)
    vols = np.linspace(0,100,21); temps = [sum(c['obj'].get_temperature_at_volume(v)*c['proportion_vol'] for c in components) for v in vols]
    dist_b = list(zip(vols,temps)); name_b = "Mezcla ("+", ".join([f"{c['obj'].name} {c['proportion_vol']*100:.0f}%" for c in components])+")"
    if verbose: logging.info(f"Blend '{name_b}': API {api_b:.1f}, SG {sg_b:.4f}, S {sulfur_b:.4f}%wt")
    return CrudeOil(name_b, api_b, sulfur_b, dist_b, verbose, True)

class DistillationCut:
    def __init__(self, name: str, t_initial_C: float, t_final_C: float, crude_oil_ref: CrudeOil):
        self.name = name; self.t_initial_C = float(t_initial_C); self.t_final_C = float(t_final_C)
        self.crude_oil = crude_oil_ref; self.yield_vol_percent: Optional[float]=0.0; self.vabp_C: Optional[float]=None
        self.api_cut: Optional[float]=None; self.sg_cut: Optional[float]=None; self.yield_wt_percent: Optional[float]=0.0
        self.sulfur_cut_wt_percent: Optional[float]=0.0; self.sulfur_cut_ppm: Optional[float]=0.0
        self.distillation_curve: Optional[List[Tuple[float,float]]]=None
        self.is_gas_cut = name.lower().startswith("gas") and self.t_final_C<=40
    def _estimate_distillation_curve(self):
        if self.yield_vol_percent is None or self.yield_vol_percent<=1e-6 or self.t_final_C<=self.t_initial_C: return None
        vabp = self.vabp_C if self.vabp_C is not None else (self.t_initial_C+self.t_final_C)/2.0
        if self.is_gas_cut: return [(0.0,self.t_initial_C),(50.0,vabp),(100.0,self.t_final_C)]
        vols=[0,5,10,30,50,70,90,95,100.0]; temps=[]
        for v in vols:
            f=1-(1/(1+np.exp(-0.05*(v-50)))) if v<50 else 1/(1+np.exp(-0.05*(v-50)))
            t=vabp-(f*(vabp-self.t_initial_C)) if v<50 else vabp+(f*(self.t_final_C-vabp)); temps.append(float(t))
        return list(zip(vols,temps))
    def _estimate_api_cut_placeholder(self, api_sensitivity_factor: float = 7.0) -> Optional[float]: # Nombre del parámetro es api_sensitivity_factor
        if self.is_gas_cut: return 110.0
        if self.vabp_C is None: return None
        t50=self.crude_oil.t50_C if self.crude_oil.t50_C>0 else self.vabp_C
        tdiff=(self.vabp_C-t50)/100.0; api_est=self.crude_oil.api-(tdiff*api_sensitivity_factor) # Usar nombre correcto
        return max(0.1,min(api_est,120.0))
    def calculate_basic_properties(self, api_sensitivity_factor: float = 7.0): # Nombre del parámetro es api_sensitivity_factor
        if self.is_gas_cut and self.t_initial_C > -50: self.t_initial_C=min(self.t_initial_C,-40.0)
        v_init=self.crude_oil.get_volume_at_temperature(self.t_initial_C); v_fin=self.crude_oil.get_volume_at_temperature(self.t_final_C)
        self.yield_vol_percent=abs(v_fin-v_init)
        if self.yield_vol_percent is not None and self.yield_vol_percent>1e-6:
            if self.is_gas_cut: self.vabp_C=(self.t_initial_C+self.t_final_C)/2.0
            else:self.vabp_C=self.crude_oil.get_temperature_at_volume(v_init+(self.yield_vol_percent/2.0))
        else: self.vabp_C=(self.t_initial_C+self.t_final_C)/2.0
        if self.vabp_C is not None or self.is_gas_cut: self.api_cut=self._estimate_api_cut_placeholder(api_sensitivity_factor=api_sensitivity_factor) # Pasar argumento
        self.sg_cut=api_to_sg(self.api_cut)
        if self.yield_vol_percent is not None and self.sg_cut is not None and self.crude_oil.sg is not None and self.crude_oil.sg>1e-6:
            f=0.85 if self.is_gas_cut else 1.0; self.yield_wt_percent=self.yield_vol_percent*(self.sg_cut/self.crude_oil.sg)*f
        else: self.yield_wt_percent=0.0
        self.distillation_curve=self._estimate_distillation_curve()
    def set_sulfur_properties(self, s_wt:Optional[float]):
        if s_wt is not None: s_eff=min(float(s_wt)*0.3,0.1) if self.is_gas_cut else float(s_wt); self.sulfur_cut_wt_percent=s_eff; self.sulfur_cut_ppm=s_eff*10000.0
        else: self.sulfur_cut_wt_percent=0.0; self.sulfur_cut_ppm=0.0
    def get_distillation_data(self): return pd.DataFrame(self.distillation_curve,columns=["Volumen (%)","Temperatura (°C)"]) if self.distillation_curve else pd.DataFrame(columns=["Volumen (%)","Temperatura (°C)"])
    def to_dict(self): return {"Corte":self.name,"T Inicial (°C)":self.t_initial_C,"T Final (°C)":self.t_final_C,"Rend. Vol (%)":self.yield_vol_percent,
                               "Rend. Peso (%)":self.yield_wt_percent,"VABP (°C)":self.vabp_C,"API Corte":self.api_cut,"SG Corte":self.sg_cut,
                               "Azufre (%peso)":self.sulfur_cut_wt_percent,"Azufre (ppm)":self.sulfur_cut_ppm}
    def __repr__(self): return f"DistillationCut('{self.name}',YVol={self.yield_vol_percent:.2f if self.yield_vol_percent is not None else 'NA'}%,S_wt={self.sulfur_cut_wt_percent:.4f if self.sulfur_cut_wt_percent is not None else 'NA'}%)"

# --- Funciones de Cálculo de Torres ---

def calculate_distillation_cuts(crude_oil: CrudeOil, cut_definitions: List[Tuple[str, float]],
                                verbose: bool = False, api_sensitivity_factor: float = 7.0,
                                empirical_data: Optional[Dict[str, Any]] = None) -> List[DistillationCut]:
    """
    Calcula los cortes de destilación para un crudo dado, opcionalmente aplicando datos empíricos.

    Args:
        crude_oil (CrudeOil): El objeto CrudeOil a procesar.
        cut_definitions (List[Tuple[str, float]]): Lista de tuplas (nombre_corte, temp_final_C).
        verbose (bool): Si imprimir información detallada.
        api_sensitivity_factor (float): Factor para estimar API de cortes.
        empirical_data (Optional[Dict[str, Any]]): Datos empíricos a aplicar (si existen y son válidos).
                                                   Formato esperado: {'distribution_data': {'cuts': [...]}}

    Returns:
        List[DistillationCut]: Lista de objetos DistillationCut calculados.
    """
    if verbose: logging.info(f"\n--- Calculating Cuts for Feed '{crude_oil.name}' (API Sens: {api_sensitivity_factor}) ---")

    # --- Aplicación de Datos Empíricos ---
    # Verifica si hay datos empíricos válidos para aplicar
    if empirical_data and isinstance(empirical_data.get('distribution_data'), dict) and empirical_data['distribution_data'].get('cuts'):
        logging.info(f"Applying empirical cut distribution for {crude_oil.name} based on provided empirical_data.")
        # Llama a la función específica para aplicar datos empíricos
        return apply_empirical_distribution(crude_oil, cut_definitions, empirical_data, verbose, api_sensitivity_factor) # Pasar api_sensitivity_factor

    # --- Cálculo Teórico (si no hay datos empíricos) ---
    cuts_output = []
    current_t_initial_C = crude_oil.ibp_C
    for i, (cut_name, cut_t_final_C_val) in enumerate(cut_definitions):
        t_start = current_t_initial_C; t_end = float(cut_t_final_C_val)
        cut = DistillationCut(name=cut_name, t_initial_C=t_start, t_final_C=t_end, crude_oil_ref=crude_oil)
        # Calcula propiedades básicas teóricamente
        cut.calculate_basic_properties(api_sensitivity_factor=api_sensitivity_factor) # Pasar argumento
        if t_end < t_start:
            if verbose: logging.warning(f"T_final ({t_end}°C) for '{cut_name}' < T_initial ({t_start}°C). Yield set to 0.")
            cut.yield_vol_percent = 0.0; cut.yield_wt_percent = 0.0
        cuts_output.append(cut)
        current_t_initial_C = t_end

    # --- Manejo del Residuo ---
    if current_t_initial_C < crude_oil.fbp_C :
        residue_name_inferred = f"Residuo de {crude_oil.name}"
        # Evitar duplicar si el último corte definido ya es un residuo
        if cut_definitions and cut_definitions[-1][0].lower().startswith("residuo"): pass
        else:
            residue_cut = DistillationCut(name=residue_name_inferred, t_initial_C=current_t_initial_C, t_final_C=crude_oil.fbp_C, crude_oil_ref=crude_oil)
            residue_cut.calculate_basic_properties(api_sensitivity_factor=api_sensitivity_factor) # Pasar argumento
            # Ajustar rendimiento del residuo
            vol_at_residue_start = crude_oil.get_volume_at_temperature(current_t_initial_C)
            residue_cut.yield_vol_percent = 100.0 - vol_at_residue_start
            if residue_cut.sg_cut and crude_oil.sg and crude_oil.sg > 1e-6 and residue_cut.yield_vol_percent is not None:
                 residue_cut.yield_wt_percent = residue_cut.yield_vol_percent * (residue_cut.sg_cut / crude_oil.sg)
            else: residue_cut.yield_wt_percent = 0.0
            cuts_output.append(residue_cut)

    # --- Distribución de Azufre (común a ambos métodos) ---
    total_crude_sulfur = crude_oil.sulfur_total_wt_percent
    if total_crude_sulfur > 0:
        total_yield_wt = sum(c.yield_wt_percent for c in cuts_output if c.yield_wt_percent is not None)
        if total_yield_wt > 1e-6:
            # Calcular denominador para el modelo de azufre
            denominator = sum(((c.vabp_C / crude_oil.t50_C if crude_oil.t50_C > 1e-6 and c.vabp_C is not None else 1.0) * (c.yield_wt_percent if c.yield_wt_percent is not None else 0.0))
                              for c in cuts_output if c.yield_wt_percent is not None and c.yield_wt_percent > 0)
            if abs(denominator) > 1e-9:
                factor_s = (total_crude_sulfur * total_yield_wt) / denominator
                for cut in cuts_output:
                    if cut.vabp_C is not None and cut.yield_wt_percent is not None and cut.yield_wt_percent > 0:
                        vabp_ratio = cut.vabp_C / crude_oil.t50_C if crude_oil.t50_C > 1e-6 else 1.0
                        sulfur_c = factor_s * vabp_ratio
                        cut.set_sulfur_properties(max(0, min(sulfur_c, total_crude_sulfur * 10))) # Limitar azufre
                    else: cut.set_sulfur_properties(0.0)
            elif verbose:
                logging.warning(f"Sulfur model failed for {crude_oil.name}. Distributing S uniformly by weight yield.")
                # Distribución uniforme si el modelo falla
                for cut in cuts_output:
                    s_dist = total_crude_sulfur * (cut.yield_wt_percent / total_yield_wt) if cut.yield_wt_percent and cut.yield_wt_percent > 0 else 0.0
                    cut.set_sulfur_properties(s_dist)
        else:
            # Si no hay rendimiento en peso, no hay azufre en los cortes
            for cut in cuts_output: cut.set_sulfur_properties(0.0)
    else:
        # Si el crudo no tiene azufre, los cortes tampoco
        for cut in cuts_output: cut.set_sulfur_properties(0.0)
    return cuts_output


# --- CORRECCIÓN: Añadir el parámetro 'empirical_data_for_crude' a la definición ---
def calculate_atmospheric_cuts(crude_oil_feed: CrudeOil,
                               atmospheric_cut_definitions: List[Tuple[str, float]],
                               verbose: bool = False,
                               api_sensitivity_factor: float = 7.0,
                               empirical_data_for_crude: Optional[Dict[str, Any]] = None # <--- PARÁMETRO AÑADIDO
                               ) -> Tuple[List[DistillationCut], Optional[DistillationCut]]:
    """
    Calcula los productos de la torre de destilación atmosférica.

    Args:
        crude_oil_feed (CrudeOil): Alimentación a la torre.
        atmospheric_cut_definitions (List[Tuple[str, float]]): Definiciones de los cortes.
        verbose (bool): Imprimir logs detallados.
        api_sensitivity_factor (float): Factor para estimar API.
        empirical_data_for_crude (Optional[Dict[str, Any]]): Datos empíricos para aplicar (si existen).

    Returns:
        Tuple[List[DistillationCut], Optional[DistillationCut]]:
            - Lista de destilados atmosféricos.
            - El residuo atmosférico (o None si no se genera).
    """
    if verbose: logging.info(f"\n--- Calculating Atmospheric Cuts for '{crude_oil_feed.name}' ---")

    # Llama a la función general de cálculo de cortes, pasando los datos empíricos
    all_atmospheric_products = calculate_distillation_cuts(
        crude_oil=crude_oil_feed,
        cut_definitions=atmospheric_cut_definitions,
        verbose=verbose,
        api_sensitivity_factor=api_sensitivity_factor,
        empirical_data=empirical_data_for_crude # <--- Pasar el argumento recibido
    )

    atmospheric_distillates = []
    atmospheric_residue_object = None

    # Lógica para separar destilados y residuo (sin cambios necesarios aquí)
    if all_atmospheric_products:
        # Caso 1: El último corte definido es explícitamente un residuo
        if atmospheric_cut_definitions and atmospheric_cut_definitions[-1][0].lower().startswith("residuo atmosf"):
            # Buscar el objeto residuo en la lista de productos generados
            found_residue = False
            temp_distillates = []
            for prod in all_atmospheric_products:
                if prod.name.lower().startswith("residuo atmosf"):
                    atmospheric_residue_object = prod
                    found_residue = True
                else:
                    temp_distillates.append(prod)
            atmospheric_distillates = temp_distillates
            if not found_residue:
                 logging.warning(f"Residuo Atmosférico definido ('{atmospheric_cut_definitions[-1][0]}') pero no encontrado claramente en productos para {crude_oil_feed.name}. Se asume que el último producto es el residuo si existe.")
                 # Si aun así no se encontró pero hay productos, asumir que el último es el residuo
                 if all_atmospheric_products:
                     atmospheric_residue_object = all_atmospheric_products[-1]
                     atmospheric_distillates = all_atmospheric_products[:-1]

        # Caso 2: El último producto generado implícitamente es un residuo
        elif all_atmospheric_products[-1].name.startswith("Residuo de"):
            atmospheric_residue_object = all_atmospheric_products[-1]
            atmospheric_distillates = all_atmospheric_products[:-1]

        # Caso 3: No hay un residuo claro (posiblemente todos los cortes definidos cubren hasta el FBP)
        else:
            # Si el número de productos coincide con las definiciones, no hubo residuo implícito
            if len(all_atmospheric_products) == len(atmospheric_cut_definitions):
                 logging.warning(f"No se generó un residuo atmosférico separado para {crude_oil_feed.name}. Verifique las temperaturas de corte.")
                 atmospheric_distillates = all_atmospheric_products
                 atmospheric_residue_object = None
            # Si hay un producto extra, es probable que sea el residuo implícito
            elif len(all_atmospheric_products) > len(atmospheric_cut_definitions):
                 atmospheric_residue_object = all_atmospheric_products[-1]
                 atmospheric_distillates = all_atmospheric_products[:-1]
                 logging.info(f"Residuo atmosférico implícito '{atmospheric_residue_object.name}' generado.")
            # Caso raro: menos productos que definiciones (podría indicar error previo)
            else:
                 atmospheric_distillates = all_atmospheric_products
                 logging.warning(f"Menos productos generados ({len(all_atmospheric_products)}) que cortes definidos ({len(atmospheric_cut_definitions)}) para {crude_oil_feed.name}.")

    return atmospheric_distillates, atmospheric_residue_object


def create_vacuum_feed_from_residue(original_crude: CrudeOil,
                                    atmospheric_residue_cut: Optional[DistillationCut],
                                    verbose: bool = False) -> Optional[CrudeOil]:
    if atmospheric_residue_cut is None or atmospheric_residue_cut.yield_vol_percent is None or atmospheric_residue_cut.yield_vol_percent <= 1e-6:
        if verbose: logging.info("Residuo atmosférico nulo o con rendimiento cero. No se crea alimentación a vacío.")
        return None
    if atmospheric_residue_cut.api_cut is None or atmospheric_residue_cut.sulfur_cut_wt_percent is None:
        logging.error(f"No se puede crear alim. vacío desde '{atmospheric_residue_cut.name}', faltan API o Azufre."); return None
    t_start_res_abs = atmospheric_residue_cut.t_initial_C; t_end_res_abs = original_crude.fbp_C
    vol_start_res_orig = original_crude.get_volume_at_temperature(t_start_res_abs)
    total_vol_res_orig_scale = 100.0 - vol_start_res_orig
    if total_vol_res_orig_scale <= 1e-6:
        if verbose: logging.info(f"Volumen de residuo atm. ({total_vol_res_orig_scale}%) muy pequeño. No se crea alim. vacío."); return None
    vac_feed_tbp = []
    for new_vol_pct in np.linspace(0, 100, 21):
        orig_equiv_vol = vol_start_res_orig + (new_vol_pct / 100.0) * total_vol_res_orig_scale
        orig_equiv_vol = min(orig_equiv_vol, 100.0)
        temp_new_vol = original_crude.get_temperature_at_volume(orig_equiv_vol)
        vac_feed_tbp.append((new_vol_pct, temp_new_vol))
    if vac_feed_tbp:
        vac_feed_tbp[0] = (0.0, t_start_res_abs)
        vac_feed_tbp[-1] = (100.0, t_end_res_abs)
    vac_feed_name = f"Alim. Vacío de {original_crude.name}"
    if verbose: logging.info(f"Creando alim. vacío: {vac_feed_name}. API: {atmospheric_residue_cut.api_cut:.1f}, S: {atmospheric_residue_cut.sulfur_cut_wt_percent:.4f}%")
    return CrudeOil(name=vac_feed_name, api_gravity=atmospheric_residue_cut.api_cut,
                    sulfur_content_wt_percent=atmospheric_residue_cut.sulfur_cut_wt_percent,
                    distillation_data_percent_vol_temp_C=vac_feed_tbp, verbose=verbose, is_blend=original_crude.is_blend) # Propagar si la original era mezcla

def calculate_vacuum_cuts(vacuum_feed: CrudeOil,
                          vacuum_cut_definitions: List[Tuple[str, float]],
                          verbose: bool = False,
                          api_sensitivity_factor: float = 7.0
                          ) -> List[DistillationCut]:
    if verbose: logging.info(f"\n--- Calculating Vacuum Cuts for '{vacuum_feed.name}' ---")
    # Los cortes de vacío generalmente se calculan teóricamente,
    # no se suelen aplicar datos empíricos de crudo aquí.
    vacuum_products = calculate_distillation_cuts(
        crude_oil=vacuum_feed, cut_definitions=vacuum_cut_definitions,
        verbose=verbose, api_sensitivity_factor=api_sensitivity_factor,
        empirical_data=None # No pasar datos empíricos aquí
    )
    return vacuum_products

def apply_empirical_distribution(crude_oil: CrudeOil, cut_definitions: List[Tuple[str, float]],
                                 empirical_data: Dict[str, Any], verbose: bool = False,
                                 api_sensitivity_factor: float = 7.0) -> List[DistillationCut]: # Añadir api_sensitivity_factor
    """
    Aplica una distribución de cortes empírica guardada a un crudo.
    Sobrescribe los cálculos teóricos para los cortes definidos en los datos empíricos.
    Calcula teóricamente los cortes no presentes en los datos empíricos.

    Args:
        crude_oil (CrudeOil): El objeto CrudeOil.
        cut_definitions (List[Tuple[str, float]]): Definiciones teóricas de cortes (nombre, T final).
        empirical_data (Dict[str, Any]): Datos empíricos {'distribution_data': {'cuts': [...]}}.
        verbose (bool): Imprimir logs.
        api_sensitivity_factor (float): Factor para cálculos teóricos de cortes no empíricos.

    Returns:
        List[DistillationCut]: Lista de objetos DistillationCut (mezcla de empíricos y teóricos).
    """
    if verbose: logging.info(f"Applying empirical distribution for {crude_oil.name}")
    cuts_output = []
    empirical_cuts_list = empirical_data.get('distribution_data', {}).get('cuts', [])

    # Crear un mapeo de nombre de corte empírico a sus datos para búsqueda rápida
    empirical_cuts_map = {cut_data.get("Corte"): cut_data for cut_data in empirical_cuts_list if isinstance(cut_data, dict) and cut_data.get("Corte")}

    def _safe_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
        """Convierte de forma segura a float, manejando None y 'N/A'."""
        if value is None or (isinstance(value, str) and value.strip().upper() in ["N/A", "NA", ""]): return default
        try: return float(value)
        except (ValueError, TypeError):
            logging.warning(f"Empirical apply: Could not convert '{value}' to float, using default {default}.")
            return default

    current_t_initial_C = crude_oil.ibp_C
    total_empirical_yield_vol = 0.0
    processed_empirical_cuts = set()

    # Iterar sobre las definiciones de corte TEÓRICAS para mantener el orden y T inicial/final
    for cut_name, cut_t_final_C_val in cut_definitions:
        t_start = current_t_initial_C; t_end = float(cut_t_final_C_val)
        cut = DistillationCut(name=cut_name, t_initial_C=t_start, t_final_C=t_end, crude_oil_ref=crude_oil)

        # Si este corte existe en los datos empíricos
        if cut_name in empirical_cuts_map:
            emp_cut = empirical_cuts_map[cut_name]
            if verbose: logging.info(f"  Using empirical data for cut '{cut_name}'")

            # Sobrescribir propiedades con datos empíricos
            cut.yield_vol_percent = _safe_float(emp_cut.get("Rend. Vol (%)"))
            cut.yield_wt_percent = _safe_float(emp_cut.get("Rend. Peso (%)"))
            cut.api_cut = _safe_float(emp_cut.get("API Corte"), None) # Permitir None si no está
            cut.sg_cut = api_to_sg(cut.api_cut) # Recalcular SG desde API empírico
            cut.vabp_C = _safe_float(emp_cut.get("VABP (°C)"), (t_start + t_end) / 2.0) # Usar VABP empírico o estimar
            cut.set_sulfur_properties(_safe_float(emp_cut.get("Azufre (%peso)"))) # Usar azufre empírico
            cut.distillation_curve = cut._estimate_distillation_curve() # Estimar curva basada en T y VABP

            total_empirical_yield_vol += cut.yield_vol_percent if cut.yield_vol_percent is not None else 0.0
            processed_empirical_cuts.add(cut_name)

        else:
            # Si el corte NO está en los datos empíricos, calcularlo teóricamente
            if verbose: logging.info(f"  No empirical data for cut '{cut_name}', calculating theoretically.")
            cut.calculate_basic_properties(api_sensitivity_factor=api_sensitivity_factor) # Usar cálculo teórico

        cuts_output.append(cut)
        current_t_initial_C = t_end # Actualizar T inicial para el siguiente corte

    # --- Manejo del Residuo (si aplica) ---
    # Si la T final del último corte definido es menor que el FBP del crudo
    if current_t_initial_C < crude_oil.fbp_C:
        residue_name_inferred = f"Residuo de {crude_oil.name}"
        # Comprobar si el residuo ya fue definido explícitamente o está en los datos empíricos
        is_residue_defined = (cut_definitions and cut_definitions[-1][0].lower().startswith("residuo")) or \
                             any(name.lower().startswith("residuo") for name in empirical_cuts_map if name not in processed_empirical_cuts)

        if not is_residue_defined:
             # Si no hay residuo definido/empírico, calcularlo teóricamente
             residue_cut = DistillationCut(name=residue_name_inferred, t_initial_C=current_t_initial_C, t_final_C=crude_oil.fbp_C, crude_oil_ref=crude_oil)
             residue_cut.calculate_basic_properties(api_sensitivity_factor=api_sensitivity_factor)
             # Ajustar rendimiento del residuo teórico
             vol_at_residue_start = crude_oil.get_volume_at_temperature(current_t_initial_C)
             residue_cut.yield_vol_percent = 100.0 - vol_at_residue_start
             if residue_cut.sg_cut and crude_oil.sg and crude_oil.sg > 1e-6 and residue_cut.yield_vol_percent is not None:
                 residue_cut.yield_wt_percent = residue_cut.yield_vol_percent * (residue_cut.sg_cut / crude_oil.sg)
             else: residue_cut.yield_wt_percent = 0.0
             cuts_output.append(residue_cut)
             if verbose: logging.info(f"  Calculated theoretical residue '{residue_cut.name}'")

    # Advertir sobre cortes empíricos no utilizados (podría indicar nombres inconsistentes)
    unused_empirical_cuts = set(empirical_cuts_map.keys()) - processed_empirical_cuts
    if unused_empirical_cuts:
        logging.warning(f"Empirical cuts defined but not matched to theoretical definitions: {', '.join(unused_empirical_cuts)}")

    # La distribución de azufre se realiza en la función llamadora (calculate_distillation_cuts)
    # después de que esta función devuelva la lista de cortes.
    return cuts_output


# --- Código de Prueba (sin cambios) ---
if __name__ == "__main__":
    crude1_data_points = [(0, 50), (10, 120), (30, 220), (50, 300), (70, 380), (90, 480), (100, 550)]
    test_crude_obj = CrudeOil(name="Crudo Ejemplo Test", api_gravity=35.0,
                              sulfur_content_wt_percent=0.10,
                              distillation_data_percent_vol_temp_C=crude1_data_points,
                              verbose=True)
    atm_cut_defs = [("Nafta Ligera", 90.0), ("Nafta Pesada", 175.0), ("Kerosene", 230.0), ("Gasóleo Atmosférico", 350.0)]
    logging.info("\n--- Prueba Torre Atmosférica ---")
    # Pasar api_sensitivity_factor explícitamente si no es el default
    atm_distillates, atm_residue = calculate_atmospheric_cuts(test_crude_obj, atm_cut_defs, verbose=True, api_sensitivity_factor=7.0)
    for c in atm_distillates:
        logging.info(f"  Distillate: {c.name}: Rend Vol {c.yield_vol_percent:.2f}%, API {c.api_cut:.1f if c.api_cut else 'N/A'}")
    if atm_residue:
        logging.info(f"  Residue: {atm_residue.name}: Rend Vol {atm_residue.yield_vol_percent:.2f}%, API {atm_residue.api_cut:.1f if atm_residue.api_cut else 'N/A'}")
        logging.info("\n--- Creando Alimentación a Vacío ---")
        vacuum_feed = create_vacuum_feed_from_residue(test_crude_obj, atm_residue, verbose=True)
        if vacuum_feed:
            vac_cut_defs = [("VGO Ligero", 450.0), ("VGO Pesado", 550.0)]
            logging.info("\n--- Prueba Torre de Vacío ---")
            vacuum_products = calculate_vacuum_cuts(vacuum_feed, vac_cut_defs, verbose=True, api_sensitivity_factor=7.0)
            for vp in vacuum_products:
                 yield_on_original_crude = (vp.yield_vol_percent / 100.0) * (atm_residue.yield_vol_percent / 100.0) * 100.0 if vp.yield_vol_percent and atm_residue.yield_vol_percent else 0.0
                 logging.info(f"  Vac Product: {vp.name}: Rend Vol (on VacFeed) {vp.yield_vol_percent:.2f}%, Rend Vol (on OrigCrude) {yield_on_original_crude:.2f}%, API {vp.api_cut:.1f if vp.api_cut else 'N/A'}")
    else:
        logging.info("No se generó residuo atmosférico para procesar en vacío.")
