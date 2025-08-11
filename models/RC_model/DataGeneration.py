import h5py as h5
import logging
import sys
import torch
from os.path import dirname as up
import numpy as np
import pandas as pd
import Physicals
import copy
from dantro._import_tools import get_from_module
import os
import random

sys.path.append(up(up(up(__file__))))

from dantro._import_tools import import_module_from_path

base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")
print(up(up(up(__file__))))

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------------------------------
# Data loading and generation utilities
# ----------------------------------------------------------------------------------------------------------------------


def apply_controller(cfg, data, heatPower_index):
    if cfg["controller"] == "TwoPointControl":
        if data[0] < cfg["T_min"]:
            heatPower = cfg["maxHeatingPower"]
        elif data[0] >= cfg["T_max"]:
            heatPower = 0
        else:
            heatPower = data[heatPower_index]
    elif cfg["controller"] == "PControl":
        heatPower = int(data[0]<=cfg["T_max"])*(cfg["T_max"]-data[0])*cfg["maxHeatingPower"]/(cfg["T_max"]-cfg["T_min"])
        if heatPower > cfg["maxHeatingPower"]:
            heatPower = cfg["maxHeatingPower"]
    elif cfg["controller"] == "RampUp":
        print(data)
        heatPower = data[heatPower_index] + cfg["maxHeatingPower"]/cfg["num_steps"]
    return heatPower


def generate_weather_based_data(cfg: dict, *, dt: float) -> torch.Tensor:
    """ Function that generates weather data based time series of length num_steps for T_in, T_out, Q_H, Q_O.

    :param cfg: configuration of data settings
    :param dt: time differential for the numerical solver (Euler in this case)
    :return: torch.Tensor of the time series for T_in, T_out, Q_H, Q_O. Tensor has shape (4, num_steps)
    """

    # Draw an initial condition for the data using the prior defined in the config

    wdata, _ = read_mos(up(up(up(__file__))) + "/data/RC_model/weatherData/Munich_5years.mos")
    wdata = wdata.to_numpy(dtype = float)

    pred_heat = False
    if "heating_data" in cfg:
        heat_data = np.array(pd.read_csv(cfg["heating_data"])["heatPower"])
        pred_heat = True

    model = get_from_module(Physicals, name = cfg["model_type"])()
    initial_condition = torch.Tensor(model.initial_condition(cfg, wdata))

    parameters = [cfg[param] for param in model.parameter_names]
    if isinstance(parameters[0], int) or isinstance(parameters[0], float):
        parameters = [[parameters]]
    out = []

    # Generate some synthetic time series
    for params in zip(*parameters):
        data = []
        densities = initial_condition[:model.dynamic_variables]
        heatPower = 0
        if isinstance(params[0], list):
            params = params[0]
        for i in range(cfg['num_steps'] + 1):

            dat = [wdata[int(i*dt/3600)][2]+273.15, heatPower, wdata[int(i*dt/3600)][8]*cfg["effWinArea"]]

            # Solve the equation for T_in and generate a time series for T_out and the Q values dt/C*((T_in-T_out)/R + QH + QO)
            if pred_heat:
                heatPower = heat_data[i]
            else:
                heatPower = apply_controller(cfg, torch.cat((densities, torch.tensor(dat))), list(model.plot_args).index("heatPower [kW]"))

            # these format acrobatics are done to use the same step function in the NN and here
            dat = [wdata[int(i*dt/3600)][2]+273.15, heatPower, wdata[int(i*dt/3600)][8] if cfg["store_raw_QSolar"] else wdata[int(i*dt/3600)][8]*cfg["effWinArea"]]
     
            data.append(torch.cat((
                torch.Tensor(densities).float(),
                torch.Tensor(dat).float()
                )))
            densities = data[-1][:model.dynamic_variables] + model.step(densities, dat, params, dt)
        out.append(torch.reshape(torch.stack(copy.deepcopy(data)), (len(data), len(model.plot_args), 1)))
    return torch.reshape(torch.stack(out), (len(out), len(out[0]), len(model.plot_args), 1))


def get_RC_circuit_data(*, data_cfg: dict, h5group: h5.Group):
    """Returns the training data for the RC_circuit model. If a directory is passed, the
    data is loaded from that directory (csv output file from BuildA). Otherwise, synthetic training data is generated
    by iteratively solving the temporal ODE system.

    :param data_cfg: dictionary of config keys
    :param h5group: h5.Group to write the training data to
    :return: torch.Tensor of training data

    """
    if "load_from_dir" in data_cfg.keys():
        if not isinstance(data_cfg["load_from_dir"]["path"], list):
            if data_cfg["load_from_dir"]["path"].endswith(".csv"):
                data_cfg["load_from_dir"]["path"] = [data_cfg["load_from_dir"]["path"]]
            else:
                rootpath = data_cfg["load_from_dir"]["path"]
                data_cfg["load_from_dir"]["path"] = []
                for root, dirs, files in os.walk(rootpath):
                    for file in files:
                        if file.startswith("_"):
                            log.debug(f"found file {os.path.join(root, file)}")
                            data_cfg["load_from_dir"]["path"].append(os.path.join(root, file))
                random.shuffle(data_cfg["load_from_dir"]["path"])
                log.info(f"\tDetected folder as dirpath, loading {len(data_cfg["load_from_dir"]["path"])} csvs that start with \"_\".")

        out = []
        for path in data_cfg["load_from_dir"]["path"]:
            with open(path, "r") as f:
                df = pd.read_csv(f)
                keys = data_cfg["load_from_dir"]["csv_keys"]
                data = torch.from_numpy(np.array([[df[keys[0]],
                                                    df[keys[1]],
                                                    df[keys[2]],
                                                    np.array(df[keys[3]])*data_cfg["load_from_dir"]["effWinArea"]]]
                                                    )).float()[:, :, :data_cfg["load_from_dir"]["subset"]].T.unsqueeze(0)
            out.append(data)
        data = torch.cat(out, axis  = 0)
        #data_cfg["dt"] = 900 #since BuildA calculates in quarterhourly steps
        attributes = data_cfg["load_from_dir"]["csv_keys"]

    elif "synthetic_data" in data_cfg.keys() and not "load_from_dir" in data_cfg.keys():

        data_cfg["synthetic_data"]["model_type"] = data_cfg["model_type"]
        data = generate_weather_based_data(
            cfg=data_cfg["synthetic_data"],
            dt=data_cfg["dt"]
        )
        attributes = list(get_from_module(Physicals, name = data_cfg["model_type"])().plot_args.keys())
    
    else:
        raise ValueError(
            f"You must supply one of 'load_from_dir' or 'synthetic data' keys!"
        )


    # Store the synthetically generated data in an h5 file
    dset = h5group.create_dataset(
        "RC_data",
        data.shape,
        maxshape=data.shape,
        chunks=True,
        compression=3,
        dtype=float,
    )

    dset.attrs["dim_names"] = ["permut", "time", "kind", "dim_name__0"]
    dset.attrs["coords_mode__time"] = "trivial"
    dset.attrs["coords_mode__kind"] = "values"
    dset.attrs["coords__kind"] = attributes
    dset.attrs["coords_mode__dim_name__0"] = "trivial"

    dset[:, :] = data
    
    
    return data

def read_mos(filename):

    print(f"Reading reference file {filename} for data generation")
    with open(filename, "r") as f:
        data = f.read()

    # scans for header and data and splits it
    n_cols = int(data[data.find(",")+1: data.find(")")])  #start of header contains "*double tab1(rows,cols)\n*"
    last_header_line = data.find(f"C{n_cols}")
    header_end = data[last_header_line:].find("\n")+last_header_line
    header = data[:header_end+1] 
    dat = data[header_end+1:]

    #converts the data to a numpy array
    arr1 = dat.split("\n")[:-1]
    arr2 = np.array([i.split("\t") for i in arr1])
    header_shape = (int(header[header.find("(")+1:header.find(",")]), int(header[header.find(",")+1:header.find(")")]))
    if arr2.shape != header_shape:
        print(f"ERROR while reading .mos file: list dimensions {arr2.shape} do not match header {header_shape}!")
        exit()
    df = pd.DataFrame(arr2, columns = [f"C{i+1}" for i in range(n_cols)])
    return df, header

#print(get_RC_circuit_data(data_cfg = {"load_from_dir": {"path": "C:/Users/Timo/Documents/SublimeProjects/NeuralABMBA/data/RC_model/MunichPI.csv", "effWinArea": 1.5, 'subset': 200}}, h5group = None).shape)

#cfg = {"load_from_dir": {"path": "C:/Users/Timo/Documents/SublimeProjects/NeuralABMBA/data/RC_model/generalization_data", "csv_keys": ["thermalZone.TAir", "weaBus.TDryBul", "totalHeatingPower.y", "weaDat.weaBus.HGloHor"], "effWinArea": 1, "subset": 2000}}
#data = get_RC_circuit_data(data_cfg = cfg, h5group = None)