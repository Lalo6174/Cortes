import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmpiricalDistributionManager:
    def __init__(self, data_dir="empirical_data"):
        self.data_dir = data_dir
        self.file_path = os.path.join(self.data_dir, "empirical_scenarios.json") # Changed filename
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logging.info(f"Created data directory: {data_dir}")
        self.all_scenarios_data = self._load_all_scenario_data()

    def _load_all_scenario_data(self) -> dict:
        """Carga todos los datos de escenarios (crudos y mezclas) guardados."""
        scenarios_data = {}
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    scenarios_data = json.load(f)
                logging.info(f"Successfully loaded data from {self.file_path}")
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON from {self.file_path}: {e}")
            except Exception as e:
                logging.error(f"Error loading scenario data from {self.file_path}: {e}")
        else:
            logging.info(f"Scenario file not found at {self.file_path}. Initializing with empty data.")
        return scenarios_data

    def _save_to_file(self) -> None:
        """Guarda todos los datos de escenarios en el archivo."""
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.all_scenarios_data, f, indent=2, ensure_ascii=False)
            logging.info(f"Successfully saved data to {self.file_path}")
        except Exception as e:
            logging.error(f"Error saving scenario data to {self.file_path}: {e}")

    def save_scenario(self, primary_key: str, scenario_type: str, scenario_name: str, distribution_data: Dict[str, Any]) -> None:
        """
        Guarda la distribución empírica para un escenario (crudo o mezcla).

        Args:
            primary_key (str): Clave principal (ej. "CrudoX_30.5" para crudo, hash para mezcla).
            scenario_type (str): Tipo de escenario ("crude" o "blend").
            scenario_name (str): Nombre descriptivo del escenario (ej. "Escenario A", "Mezcla Optima X").
            distribution_data (dict): Datos de la distribución (cortes, propiedades, componentes si es mezcla).
        """
        timestamp = datetime.now().isoformat()

        if primary_key not in self.all_scenarios_data:
            self.all_scenarios_data[primary_key] = {
                "scenario_type": scenario_type,
                "scenarios": {}
            }
        elif self.all_scenarios_data[primary_key]["scenario_type"] != scenario_type:
            # This case should ideally be handled or prevented by UI logic
            logging.warning(f"Primary key {primary_key} already exists with type "
                            f"{self.all_scenarios_data[primary_key]['scenario_type']}, "
                            f"but attempting to save as {scenario_type}. Overwriting type is not recommended.")
            self.all_scenarios_data[primary_key]["scenario_type"] = scenario_type # Or raise error

        # Ensure 'distribution_data' always includes 'scenario_type' for clarity
        distribution_data_with_type = distribution_data.copy()
        distribution_data_with_type['scenario_type'] = scenario_type # Add/overwrite for explicitness

        self.all_scenarios_data[primary_key]["scenarios"][scenario_name] = {
            "distribution_data": distribution_data_with_type, # Store the modified dict
            "last_updated": timestamp
        }
        logging.info(f"Saved scenario '{scenario_name}' for primary key '{primary_key}' (type: {scenario_type}).")
        self._save_to_file()

    def get_scenario_data(self, primary_key: str, scenario_name: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene los datos de un escenario específico.

        Args:
            primary_key (str): Clave principal del crudo o mezcla.
            scenario_name (str): Nombre del escenario a obtener.

        Returns:
            dict: Datos de la distribución o None si no se encuentra.
        """
        if primary_key not in self.all_scenarios_data:
            logging.warning(f"Primary key '{primary_key}' not found.")
            return None
        if scenario_name not in self.all_scenarios_data[primary_key]["scenarios"]:
            logging.warning(f"Scenario '{scenario_name}' not found for primary key '{primary_key}'.")
            return None
        
        # Return the 'distribution_data' part which contains the actual scenario payload
        return self.all_scenarios_data[primary_key]["scenarios"][scenario_name]["distribution_data"]

    def list_all_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """
        Devuelve todos los escenarios guardados, agrupados por su clave primaria.
        """
        return self.all_scenarios_data

    def list_all_scenarios_flat(self) -> List[Dict[str, Any]]:
        """
        Devuelve una lista plana de todos los escenarios, cada uno con su clave primaria y nombre.
        Útil para selectores en la UI.
        """
        flat_list = []
        for primary_key, data in self.all_scenarios_data.items():
            scenario_type = data.get("scenario_type", "unknown")
            for scenario_name, scenario_content in data.get("scenarios", {}).items():
                flat_list.append({
                    "primary_key": primary_key,
                    "scenario_name": scenario_name,
                    "scenario_type": scenario_type,
                    "last_updated": scenario_content.get("last_updated"),
                    "distribution_data": scenario_content.get("distribution_data", {}) # Include for better display names
                })
        return flat_list


    def get_scenarios_for_primary_key(self, primary_key: str) -> List[Tuple[str, str]]:
        """
        Obtiene todos los nombres de escenarios y fechas de actualización para una clave primaria dada.
        """
        if primary_key not in self.all_scenarios_data:
            return []
        return [
            (name, details["last_updated"])
            for name, details in self.all_scenarios_data[primary_key]["scenarios"].items()
        ]

    def delete_scenario(self, primary_key: str, scenario_name: Optional[str] = None) -> bool:
        """
        Elimina un escenario específico o todos los escenarios bajo una clave primaria.

        Args:
            primary_key (str): Clave principal del crudo o mezcla.
            scenario_name (str, optional): Nombre del escenario a eliminar.
                                       Si es None, elimina la clave primaria y todos sus escenarios.

        Returns:
            bool: True si se eliminó exitosamente, False si no.
        """
        if primary_key not in self.all_scenarios_data:
            logging.warning(f"Attempted to delete non-existent primary key: {primary_key}")
            return False

        if scenario_name is None: # Delete all scenarios for this primary key
            del self.all_scenarios_data[primary_key]
            logging.info(f"Deleted all scenarios for primary key: {primary_key}")
            self._save_to_file()
            return True
        else: # Delete specific scenario
            if scenario_name in self.all_scenarios_data[primary_key]["scenarios"]:
                del self.all_scenarios_data[primary_key]["scenarios"][scenario_name]
                logging.info(f"Deleted scenario '{scenario_name}' for primary key: {primary_key}")
                # If no scenarios are left under this primary key, remove the primary key itself
                if not self.all_scenarios_data[primary_key]["scenarios"]:
                    del self.all_scenarios_data[primary_key]
                    logging.info(f"Removed empty primary key: {primary_key} after deleting last scenario.")
                self._save_to_file()
                return True
            else:
                logging.warning(f"Scenario '{scenario_name}' not found for deletion under primary key: {primary_key}")
                return False
