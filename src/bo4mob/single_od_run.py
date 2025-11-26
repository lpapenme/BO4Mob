# Standard library imports
import os
import pprint
import sys
import warnings
from importlib.resources import files
from pathlib import Path
from typing import List, Literal, Optional

# Third-party imports
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from botorch.exceptions import BadInitialCandidatesWarning

# Local application imports
from bo4mob.simulation.data_loader import load_config_single_od_run, od_xml_to_df
from bo4mob.simulation.evaluation import run_single_od_evaluation
from bo4mob.utils.path_utils import prepare_run_paths
from bo4mob.utils.plot_utils import save_fit_to_gt_plots_single_run

matplotlib.use("Agg")
plt.ioff()
warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)

# =====================
# SUMO Environment Setup
# =====================

# Ensure SUMO_HOME is set and tools are accessible
if "SUMO_HOME" in os.environ:
    tools_path = os.path.join(os.environ["SUMO_HOME"], "tools")
    if os.path.exists(tools_path):
        sys.path.append(tools_path)
        os.environ["LIBSUMO_AS_TRACI"] = "1"  # Optional: faster simulation
    else:
        raise FileNotFoundError(f"SUMO 'tools' directory not found at {tools_path}")
else:
    warnings.warn("SUMO_HOME environment variable not set; SUMO tools may be unavailable.")


# =====================
# Refactored Main Function
# =====================


def run_single_simulation(
        network_name: Literal["1ramp", "2corridor", "3junction", "4smallRegion", "5fullRegion"] = "1ramp",
        date: int = 221014,
        hour: Literal["06-07", "08-09", "17-18"] = "08-09",
        eval_measure: Literal["count", "speed"] = "count",
        routes_per_od: Literal["single", "multiple"] = "single",
        od_csv: Optional[str] = None,
        od_values: Optional[List[int]] = None,
) -> None:
    """
    Run a single OD simulation using SUMO programmatically.

    This function handles:
    - Loading simulation configuration, OD matrices, route data, and sensor data.
    - Running one simulation with the provided OD input (from od_values or od_csv).
    - Saving simulation results, including output flows and evaluation plots.
    - (Optionally) launching SUMO-GUI to visualize the simulation.

    Args:
        network_name: Name of the simulation network.
        date: Date for the simulation (e.g., 221014).
        hour: Time window for the simulation.
        eval_measure: Evaluation measurement type ('count' or 'speed').
        routes_per_od: Type of routes to use ('single' or 'multiple').
        od_csv: Optional path to an external OD CSV file.
        od_values: Optional list of three integer OD values for the '1ramp' network.
    """
    # =====================
    # Set experiment settings
    # =====================
    print("Running simulation with the following settings:")
    pprint.pprint(locals())

    # Get the root path of the installed package's data
    package_root = files('bo4mob')

    # =====================
    # Load configuration
    # =====================
    config = load_config_single_od_run(config_file_name=f"sim_setup_network_{network_name}.json")
    pprint.pprint(dict(config))

    # =====================
    # Load input data from within the package
    # =====================
    od_df_base = od_xml_to_df(config["od_xml"])
    dim_od = od_df_base.shape[0]
    print(f"Number of OD pairs: {dim_od}")

    routes_csv_path = config["routes_csv"]
    if routes_per_od == 'single':
        routes_csv_path = routes_csv_path.with_name("routes_single.csv")
    elif routes_per_od == 'multiple':
        routes_csv_path = routes_csv_path.with_name("routes_multiple.csv")
    routes_df = pd.read_csv(routes_csv_path, index_col=0)

    true_sensor_file_name = f"gt_link_data_{network_name}_{date}_{hour}.csv"
    sensor_data_path = package_root / 'sensor_data' / str(date) / true_sensor_file_name
    sensor_measure_gt = pd.read_csv(sensor_data_path)

    link_selection = list(map(str, sensor_measure_gt["link_id"].tolist()))
    print(f"Number of sensors: {len(link_selection)}")

    # =====================
    # Set up paths for simulation run
    # =====================
    if od_csv:
        od_file = Path(od_csv).stem
    elif network_name == "1ramp" and od_values:
        od_file = "od_" + "-".join(map(str, od_values))
    else:
        od_file = None

    path_run_detail, path_run_simul, path_run_result, path_existence = prepare_run_paths(
        config["path_run"], date, hour, eval_measure, routes_per_od, seed=None, od_file=od_file
    )

    if path_existence:
        print(f"Run already exists at {path_run_detail}. Exiting.")
        return

    # =====================
    # Load target OD values
    # =====================
    if od_csv:
        od_file_path = Path(od_csv)
        if not od_file_path.exists():
            raise FileNotFoundError(f"OD file not found: {od_file_path}")
        od_df_target = pd.read_csv(od_file_path)
        x = od_df_target["flow"].to_numpy()
        print(f"Loaded target OD values with {len(x)} flows from {od_file_path}.")
    elif network_name == "1ramp" and od_values:
        if len(od_values) != 3:
            raise ValueError("Exactly three OD values must be provided for the '1ramp' network.")
        x = od_values
        print(f"Using provided OD values for 1ramp: {x}")
    else:
        raise ValueError("Either 'od_csv' must be provided or 'od_values' must be specified for '1ramp'.")

    # =====================
    # Run simulation and visualize results
    # =====================
    curr_link_stats = run_single_od_evaluation(
        x, od_df_base, config, path_run_detail, path_run_simul,
        path_run_result, routes_df, routes_per_od, link_selection,
        eval_measure, sensor_measure_gt
    )
    save_fit_to_gt_plots_single_run(eval_measure, x, sensor_measure_gt, curr_link_stats, path_run_detail, network_name)
