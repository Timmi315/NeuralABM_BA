import logging
import copy
from typing import Sequence, Union
import sys

import xarray as xr
from os.path import dirname as up
import os

from utopya.eval import PlotHelper, is_plot_func
from dantro.plot.funcs.generic import make_facet_grid_plot

from utopya.eval import PlotHelper, is_plot_func
from utopya.eval import is_operation

from dantro._import_tools import import_module_from_path, get_from_module

import numpy as np
from torchdiffeq import odeint
import torch
import itertools

sys.path.append(up(up(up(__file__)))+"\\models\\RC_model")

Physicals = import_module_from_path(mod_path=up(up(up(__file__)))+"\\models\\RC_model", mod_str="RC_model")



log = logging.getLogger(__name__)

@is_plot_func()
def plot_model_prediction(
    ds: xr.Dataset,
    hlpr: PlotHelper,
    *,
    #_is_facetgrid: bool,
    x: str,
    y: str,
    yerr: str = None,
    hue: str = None,
    label: str = None,
    add_legend: bool = True,
    smooth_kwargs: dict = {},
    linestyle: Union[str, Sequence] = "solid",
    **plot_kwargs,
):
    physical = get_from_module(Physicals, name = ds["multiverse"]["0"]["data"]["RC_model"]["parameters"].attrs["model_type"])()
    #print(ds["multiverse"]["0"]["data"]["RC_model"]["loss"][:])
    loss = ds["multiverse"]["0"]["data"]["RC_model"]["loss"][:]
    #print(ds["multiverse"]["0"]["data"]["RC_model"]["parameters"].coords)
    #print(ds["multiverse"]["0"]["data"]["RC_model"].attrs["model_type"])
    #print(ds["multiverse"]["0"]["data"]["RC_model"]["RC_data"].values)
    #print(physical.parameter_names)
    #print(ds["multiverse"]["0"]["data"]["RC_model"]["RC_data"].coords["kind"])

    loss_min_index = 0
    for index, l in enumerate(loss):
        if l < loss[loss_min_index]:
            loss_min_index = index

    best_params = ds["multiverse"]["0"]["data"]["RC_model"]["parameters"][loss_min_index]

    

    #print("best_params", best_params)

    physical.set_params(torch.Tensor(best_params.values))

    physical.dt = ds["multiverse"]["0"]["data"]["RC_model"]["parameters"].attrs["dt"]
    time = torch.arange(0, len(ds["multiverse"]["0"]["data"]["RC_model"]["RC_data"][0, :,0,:]), 1.0)

    physical.external_data = torch.Tensor(ds["multiverse"]["0"]["data"]["RC_model"]["RC_data"].values)[0, :,physical.dynamic_variables:,:].squeeze()

    #print("======physical input shapes =========")
    #print(physical.dynamic_variables)

    #print(type(torch.Tensor(ds["multiverse"]["0"]["data"]["RC_model"]["RC_data"].values)[0, 0,:physical.dynamic_variables,0]))

    #print(type(physical.external_data))

    #print(type(time))

    sim = odeint(physical,
                torch.Tensor(ds["multiverse"]["0"]["data"]["RC_model"]["RC_data"].values)[0, 0,:physical.dynamic_variables,0],
                time,
                method = 'euler',
                options={'step_size': 1.0}
                )
    #print(sim)
    if sim.isnan().any():
        nan = True
    else:
        nan = False



    for i in range(physical.dynamic_variables):
        hlpr.ax.plot(time*physical.dt, ds["multiverse"]["0"]["data"]["RC_model"]["RC_data"][0, :,i,:], label = physical.data_names[i] + " (true)", alpha = 0.5)
        if not nan:
            hlpr.ax.plot(time*physical.dt, sim[:,i], label = physical.data_names[i] + " (estim.)")
    hlpr.ax.legend()
    hlpr.ax.set_ylabel("Temperature [K]")
    hlpr.ax.set_xlabel("time [s]")
    hlpr.ax.set_title(f"Simulation of the model using the parameters {[f"{physical.parameter_names[i]}: {best_params.values[i]}" for i in range(len(physical.parameter_names))]}")

@is_plot_func()
def plot_model_pred(
    x,
    hlpr: PlotHelper,
    uni,
    dt,
    **kwargs
):
    print(dt)
    print("x", x)
    print(uni)

    
@is_plot_func()
def plot_simulation_delta(
    ds: xr.Dataset,
    hlpr: PlotHelper,
    *,
    #_is_facetgrid: bool,
    x: str,
    y: str,
    yerr: str = None,
    hue: str = None,
    label: str = None,
    add_legend: bool = True,
    smooth_kwargs: dict = {},
    linestyle: Union[str, Sequence] = "solid",
    **plot_kwargs,
):
    physical = get_from_module(Physicals, name = ds["multiverse"]["0"]["data"]["RC_model"]["parameters"].attrs["model_type"])()
    #print(ds["multiverse"]["0"]["data"]["RC_model"]["loss"][:])
    loss = ds["multiverse"]["0"]["data"]["RC_model"]["loss"][:]
    #print(ds["multiverse"]["0"]["data"]["RC_model"]["parameters"].coords)
    #print(ds["multiverse"]["0"]["data"]["RC_model"].attrs["model_type"])
    #print(ds["multiverse"]["0"]["data"]["RC_model"]["RC_data"].values)
    #print(physical.parameter_names)
    #print(ds["multiverse"]["0"]["data"]["RC_model"]["RC_data"].coords["kind"])

    loss_min_index = 0
    for index, l in enumerate(loss):
        if l < loss[loss_min_index]:
            loss_min_index = index

    best_params = ds["multiverse"]["0"]["data"]["RC_model"]["parameters"][loss_min_index]

    

    #print("best_params", best_params)

    physical.set_params(torch.Tensor(best_params.values))

    physical.dt = ds["multiverse"]["0"]["data"]["RC_model"]["parameters"].attrs["dt"]
    time = torch.arange(0, len(ds["multiverse"]["0"]["data"]["RC_model"]["RC_data"][0, :,0,:]), 1.0)

    physical.external_data = torch.Tensor(ds["multiverse"]["0"]["data"]["RC_model"]["RC_data"].values)[0, :,physical.dynamic_variables:,:].squeeze()

    #print("======physical input shapes =========")
    #print(physical.dynamic_variables)

    #print(type(torch.Tensor(ds["multiverse"]["0"]["data"]["RC_model"]["RC_data"].values)[0, 0,:physical.dynamic_variables,0]))

    #print(type(physical.external_data))

    #print(type(time))

    sim = odeint(physical,
                torch.Tensor(ds["multiverse"]["0"]["data"]["RC_model"]["RC_data"].values)[0, 0,:physical.dynamic_variables,0],
                time,
                method = 'euler',
                options={'step_size': 1.0}
                )
    #print(sim)
    if sim.isnan().any():
        nan = True
    else:
        nan = False


    for i in range(physical.dynamic_variables):
        if not nan:
            hlpr.ax.plot(time*physical.dt,
                np.array(sim[:,i]) - np.array(ds["multiverse"]["0"]["data"]["RC_model"]["RC_data"][0, :,i,:].squeeze()),
                label = physical.data_names[i] + " (estim - true)"
                )
    hlpr.ax.legend()
    hlpr.ax.set_ylabel("Temperature [K]")
    hlpr.ax.set_xlabel("time [s]")
    hlpr.ax.set_title(f"Simulation of the model using the parameters {[f"{physical.parameter_names[i]}: {best_params.values[i]}" for i in range(len(physical.parameter_names))]}")

@is_operation("simulate_model")
def simulate(
    marginals: xr.DataArray,
    *args,
    model_type: str,
    dt: int,
    rc_data: xr.Dataset,
    mode: str = None,
    horizon: int = 10e10

):
    #print(f"simulate model called with {len(rc_data)} datasets and model type = {model_type}")
    # read attributes and real data from multiverse data if called from multiverse
    if isinstance(model_type, xr.DataArray):
        model_type = model_type.values.flat[0]
        
        dt = dt.values.flat[0]
    elif isinstance(model_type, list):
        dt = dt.values.flat[0]

    if not isinstance(marginals, list):
        marginals = [marginals]


    # get physical obkect for sim
    if isinstance(model_type, list):
        physical = [get_from_module(Physicals, name = mod)() for mod in model_type]
    else:
        physical = [get_from_module(Physicals, name = model_type)()]

    # remove all multiverse dimensions
    #rc_data = np.array(rc_data.values.flat)[:rc_data.sizes["time"]*rc_data.sizes["kind"]]
    #rc_data = rc_data.reshape((int(rc_data.shape[0]/len(physical.plot_args)), len(physical.plot_args)))

    if isinstance(rc_data, list):
        dims_to_reduce = [dim for dim in rc_data[0].dims if dim not in ("time", "kind")]
        rc_data = [np.array(data.isel({dim: 0 for dim in dims_to_reduce})) for data in rc_data]
    else: 
        dims_to_reduce = [dim for dim in rc_data.dims if dim not in ("time", "kind")]
        rc_data = [np.array(rc_data.isel({dim: 0 for dim in dims_to_reduce}))]
    longest_data = rc_data[np.argmax([data.shape[0] for data in rc_data])]
    
    all_sims = []

    #print("rc_shapes", [data.shape for data in rc_data])

    for i, phys in enumerate(physical):

        # read best parameters (currently from min(loss) or max(neg_exp) - might make sense to fit gauss?
        params = []
        best_params = []
        for k, key in enumerate(marginals[i].coords["parameter"].values):
            params.append(key)
            best_idx = marginals[i]["y"][k].argmax()
            best_params.append(marginals[i]["x"][k][best_idx].values.item())
        # reorder params to match the physical's order
        best_params = [best_params[params.index(param_name)] for param_name in phys.parameter_names]

        print(f"Simulating model {model_type[i]} using {[f"{phys.parameter_names[l]}: {best_params[l]}" for l in range(len(params))]} over {rc_data[i].shape[0]} steps")

        # set the physical up for simulation
        phys.set_params(torch.Tensor(best_params))
        phys.dt = dt
        phys.external_data = torch.Tensor(rc_data[i])[:,phys.dynamic_variables:]

        # set up simulation and run
        time = []
        if rc_data[i].shape[0] <= horizon:
            time.append(torch.arange(0, rc_data[i].shape[0], 1.0))
        else:
            for k in range(int((rc_data[i].shape[0])/horizon)+int((rc_data[i].shape[0])/horizon%1!=0)):
                time.append(torch.arange(k*horizon, min((k+1)*horizon, rc_data[i].shape[0]), 1.0))
        sims = []
        for t in time:
            sim = odeint(phys,
                        torch.Tensor(rc_data[i])[int(t[0]), :phys.dynamic_variables],
                        t,
                        method = 'euler',
                        options={'step_size': 1.0}
                        ).unsqueeze(0).unsqueeze(2)
            sims.append(sim)
        #print(sim)
        #print(np.isnan(sim).any())
        sim = torch.cat(sims, dim = 1)
        all_sims.append(sim)


    delta = [(sim[0, :, :physical[i].dynamic_variables, 0] - torch.Tensor(rc_data[i])[:,:physical[i].dynamic_variables])[:, 0] for i, sim in enumerate(all_sims)]
    mae = np.array([np.sum(np.abs(np.array(d))/d.shape[0]) for d in delta])
    rmse = np.array([np.sqrt((d**2).sum()/d.shape[0]) for d in delta])
    sim_out = np.array([torch.Tensor(rc_data[np.argmax([data.shape[0] for data in rc_data])][:, 0])] + [pad_arr_with_nans(sim[0, :, 0, 0], longest_data.shape[0]) for i, sim in enumerate(all_sims)]).T
    delta_out = np.array([pad_arr_with_nans(delt, longest_data.shape[0]) for delt in delta]).T
    if mode == None:
        return sim_out, delta_out, mae, rmse
    # add real data along axis 2 or compute delta depending on mode
    if mode == "sim":
        out = torch.cat([torch.Tensor(rc_data[i])[:, :1]] + [sim[0, :, :1, 0] for i, sim in enumerate(all_sims)], axis = 1)
        #print(out.shape)
        return out
    elif mode == "mae":
        delta = [(sim[0, :, :physical[i].dynamic_variables, 0] - torch.Tensor(rc_data[i])[:,:physical[i].dynamic_variables])[:, 0] for i, sim in enumerate(all_sims)]
        mae = np.array([np.sum(np.abs(np.array(d))/d.shape[0]) for d in delta])
        #print(mae)
        return mae
    else:
        delta = [(sim[0, :, :physical[i].dynamic_variables, 0] - torch.Tensor(rc_data[i])[:,:physical[i].dynamic_variables])[:, 0] for i, sim in enumerate(all_sims)]
        mae = np.array([np.ones(d.shape[0])*np.sum(np.abs(np.array(d))/d.shape[0]) for d in delta])
        return np.concatenate([delta, mae], axis = 0).T

def pad_arr_with_nans(arr, target_len):
    return np.concatenate((np.array(arr), np.full((target_len-len(arr), *arr.shape[1:]), np.nan)))

@is_operation("select_first_uni_time_coords")
def select(data):
    # select the time coords alon the right axis, this one's pretty hardcoded
    dims_to_reduce = [dim for dim in data.dims if dim not in ("time")]

    # Step 2: Index the dataset at 0 along those dims
    out = np.array(data.isel({dim: 0 for dim in dims_to_reduce}))
    return out
    #return torch.Tensor(data.values)[:,0]


@is_operation("flatten_dims_except")
#@apply_along_dim
def flatten_dims(
    ds: Union[xr.Dataset, xr.DataArray, list],
    dims_to_keep,
    dim_name,
    *,
    new_coords: Sequence = None,
) -> Union[xr.Dataset, xr.DataArray]:

    if isinstance(ds, list):
        return [flatten_dims(element, dims_to_keep, dim_name, new_coords = new_coords) for element in ds]

    #print(ds.values)
    new_dim, dims_to_stack = dim_name, [dim for dim in list(ds.dims) if dim not in dims_to_keep]
    #print(f"new_dim: {new_dim}, dims_to_stack: {dims_to_stack}")

    # Check if the new dimension name already exists. If it already exists, use a temporary name for the new dimension
    # switch back later
    _renamed = False
    if new_dim in list(ds.coords.keys()):
        new_dim = f"__{new_dim}__"
        _renamed = True

    # Stack and drop the dimensions
    ds = ds.stack({new_dim: dims_to_stack})
    #print(ds)
    q = set(dims_to_stack)
    q.add(new_dim)
    ds = ds.drop_vars(q)
    #print(ds)

    # Name the stacked dimension back to the originally intended name
    if _renamed:
        ds = ds.rename({new_dim: dim_name})
        new_dim = dim_name
    # Add coordinates to new dimension and return
    if new_coords is None:
        out =  ds.assign_coords({new_dim: np.arange(len(ds.coords[new_dim]))})
    else:
        out =  ds.assign_coords({new_dim: new_coords})
    #print(out)
    return out

@is_operation("broadcast_dims")
def broadcast_e(
    ds1: xr.DataArray, ds2: xr.DataArray, broadcast_dims: list, *, x: str = "x", p: str = "loss", **kwargs
) -> xr.Dataset:
    all_coords = list(dict.fromkeys(list(ds1.coords) + list(ds2.coords)))
    exclude_dims = list(set(all_coords) - set(broadcast_dims))

    return xr.broadcast(xr.Dataset({x: ds1, p: ds2}), exclude = exclude_dims, **kwargs)[0]

@is_operation("myprint")
def myprint(data):
    print(data)
    return data

@is_operation("dims2list")
def split_dataset_along_dim(ds, dim):
    #out = {
    #    (ds.coords[dim].isel({dim: i}).item() if dim in ds.coords else i): 
    #    ds.isel({dim: i}).drop_vars(dim, errors="ignore")
    #    for i in range(ds.sizes[dim])
    #    }
    if isinstance(ds, list):
        nested = [split_dataset_along_dim(i, dim) for i in ds]
        return [item for sublist in nested for item in sublist]
    #print("splitting ", dim)
    out = [
        drop_fully_nan_entries(ds.isel({dim: i}).drop_vars(dim, errors="ignore"))
        for i in range(ds.sizes[dim])
        ]
    return out


def drop_fully_nan_entries(data):
    """
    Iteratively drops entries from all dimensions where values are fully NaN.
    Works for both DataArray and Dataset.
    """
    for dim in data.dims:
        # Get all other dims
        reduce_dims = [d for d in data.dims if d != dim]

        if reduce_dims:  # skip reduction if scalar
            mask = ~data.isnull().all(dim=reduce_dims)
            data = data.sel({dim: data[dim][mask]})

    return data

@is_operation("get_model_types_from_multiverse")
def get_mtypes(ds, padding = 1):
    dim = "model_type"
    out = []
    for model_type in ds[dim].values:
        for _ in range(padding):
            out.append(model_type)
    #print(out)
    return out

@is_operation("cart_prod_string")
def cart_prod_string(a, b):
    return [elem[0] + elem[1] for elem in itertools.product(a, b)]


@is_operation("bar_plot_groups")
def compute_grouped_bar_positions(group_shape, base=0.8, step=0.2):
    """
    Computes bar positions for arbitrarily nested groups.
    
    Args:
        group_shape (tuple): A tuple like (4, 3, 2) specifying group sizes per level.
        base (float): Starting base position for the outermost group.
        step (float): Step size between nested group levels (affects spacing).
        
    Returns:
        List of float positions.
    """
    from itertools import product

    # Create all index combinations, e.g. (0, 1, 2)
    index_ranges = [range(n) for n in group_shape]
    index_combinations = list(product(*index_ranges))

    positions = []
    for combo in index_combinations:
        pos = 0.0
        for i, idx in enumerate(combo):
            pos += idx * (step ** (i))
        positions.append(base + pos)
    print("positions", positions)

    return positions

@is_operation("pad_array")
def pad_array(arr, padding = 1):
    out = []
    for element in arr:
        for _ in range(padding):
            out.append(element)
    return out

@is_operation("repeat_array")
def repeat_array(arr, rep = 1):
    out = []
    for _ in range(rep):
        for element in arr:
            out.append(element)
    return out

@is_operation("filename_from_path")
def filename_from_path(x):
    if (isinstance(x, list)):
        return [filename_from_path(i) for i in x]
    return os.path.split(x)[-1]
