#!/usr/bin/env python3
import sys
from os.path import dirname as up
import Physicals
import os

import coloredlogs
import h5py as h5
import numpy as np
import ruamel.yaml as yaml
import torch
from dantro import logging
from dantro._import_tools import import_module_from_path, get_from_module


sys.path.append(up(up(__file__)))
sys.path.append(up(up(up(__file__))))

RC_model = import_module_from_path(mod_path=up(up(__file__)), mod_str="RC_model")
base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")

log = logging.getLogger(__name__)
coloredlogs.install(fmt="%(levelname)s %(message)s", level="INFO", logger=log)


# ----------------------------------------------------------------------------------------------------------------------
# Performing the simulation run
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    cfg_file_path = sys.argv[1]

    log.note("   Preparing model run ...")
    log.info(f"   Loading config file:\n        {cfg_file_path}")
    yamlc = yaml.YAML(typ="safe")
    with open(cfg_file_path) as cfg_file:
        cfg = yamlc.load(cfg_file)
    model_name = cfg.get("root_model_name", "RC_model")
    log.note(f"   Model name:  {model_name}")
    model_cfg = cfg[model_name]

    # Select the training device and number of threads to use
    device = model_cfg["Training"].get("device", None)
    if device is None:
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    num_threads = model_cfg["Training"].get("num_threads", None)
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    log.info(
        f"   Using '{device}' as training device. Number of threads: {torch.get_num_threads()}"
    )

    # Get the random number generator
    log.note("   Creating global RNG ...")
    rng = np.random.default_rng(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.random.manual_seed(cfg["seed"])

    log.note(f"   Creating output file at:\n        {cfg['output_path']}")
    h5file = h5.File(cfg["output_path"], mode="w")
    h5group = h5file.create_group(model_name)

    # Get the training data
    log.info("   Fetching training data...")
    training_data = RC_model.get_RC_circuit_data(data_cfg=model_cfg["Data"], h5group=h5group).to(
        device
    )
    
    log.info(f"      training_data.shape = {training_data.shape}")

    # get the physical model object used for simulating temperature.
    physical = get_from_module(Physicals, name = model_cfg["Training"]["model_type"])()

    # Initialise the neural net
    if model_cfg["NeuralNet"]["type"] == "mlp-single": #workaround for sweeping over mlp:predict and mlp:single-input
        model_cfg["Training"]["mode"] = "single-input"
        log.info("detected mlp-sinlge key. Setting cfg.Training.mode to single-input")
    if "pretrained" in model_cfg["NeuralNet"].keys():
        pretrained_path = model_cfg["NeuralNet"]["pretrained"]
        if not pretrained_path == "None" and (os.path.split(pretrained_path)[-1].startswith(("mlp", "lstm"))): #if a net type can be inferred from filename
            model_cfg["NeuralNet"]["type"] = os.path.split(pretrained_path)[-1][:os.path.split(pretrained_path)[-1].find("_")]
            log.info(f"\tInferred net type {model_cfg["NeuralNet"]["type"]} from pretrained path {pretrained_path}.")

    if model_cfg["Training"]["mode"] == "single-input":
        input_size = physical.dynamic_variables
    elif model_cfg["NeuralNet"]["type"] == "lstm":
        input_size = training_data.shape[2]
    else: 
        input_size = model_cfg["NeuralNet"]["lookback"]*training_data.shape[2]

    log.info(f"   Initializing the {model_cfg["NeuralNet"]["type"]} in {model_cfg["Training"]["mode"]} mode (inpsize: {input_size}, outpsize {len(physical.parameter_names)}) ...")
    log.info(f"   Using physical {model_cfg["Training"]["model_type"]} and train_data.shape = {training_data.shape}")
    
    
    if model_cfg["NeuralNet"]["type"] == "optimizer":
            net = base.Optimizer(input_size=input_size,
                #input_size = 1,
                output_size=len(physical.parameter_names),
                **model_cfg["NeuralNet"],
            ).to(device)
    elif model_cfg["NeuralNet"]["type"] == "lstm":
        net = base.Lstm(input_size = input_size,
            output_size=len(physical.parameter_names),
            **model_cfg["NeuralNet"],
            )
    else:
        net = base.NeuralNet(
            input_size=input_size,
            #input_size = 1,
            output_size=len(physical.parameter_names),
            **model_cfg["NeuralNet"],
        ).to(device)

    if "pretrained" in model_cfg["NeuralNet"].keys():
        if not model_cfg["NeuralNet"]["pretrained"] == "None": #for sweeping over the "pretrained" entry, yml entries are parsed as strings
            net.load_state_dict(torch.load(model_cfg["NeuralNet"]["pretrained"], weights_only = True))
            log.info(f"\t loaded pretrained model from {model_cfg["NeuralNet"]["pretrained"]}")

    # Initialise the model
    model = RC_model.NN(
        rng=rng,
        h5group=h5group,
        neural_net=net,
        write_every=cfg["write_every"],
        write_start=cfg["write_start"],
        num_steps=training_data.shape[1],
        training_data=training_data[:, :, :physical.dynamic_variables, :],
        external_data=training_data[:, :, physical.dynamic_variables:, :],
        physical = physical,
        lookback = model_cfg["NeuralNet"]["lookback"],
        dt=model_cfg["Data"]["dt"],
        **model_cfg["Training"],
    )
    model.dset_parameters.attrs["model_type"] = model_cfg["Training"]["model_type"]
    model.dset_parameters.attrs["dt"] = model_cfg["Data"]["dt"]
    log.info(f"   Initialized model '{model_name}'.")

    num_epochs = cfg["num_epochs"]
    log.info(f"   Now commencing training for {num_epochs} epochs ...")
    log.progress(
        f"   Epochs: {num_epochs}"
        f"  \t\t\t\t\t"
        f"  Parameters: {physical.parameter_names}"
        )
    for i in range(num_epochs):
        model.epoch()
        log.progress(
            f"   Epoch {i+1} / {num_epochs}; "
            f"   Loss: {model.current_loss}"
            f"   Parameters: {model.current_predictions}"
        )

    log.info("   Simulation run finished.")
    log.info("   Wrapping up ...")

    if model_cfg["Training"]["mode"] == "generalize":
        log.info(os.path.split(cfg_file_path)[0])
        net_out_path = os.path.join(os.path.split(cfg_file_path)[0], "trained_net.pth")
        log.info(f"\t{net_out_path}")

        torch.save(net.state_dict(), net_out_path)
        log.info(f"\tSaved Neural Net to {net_out_path}")

    h5file.close()
    #torch.save(net.state_dict(), f"{cfg["output_path"][:-7]}neural_net.pth")

    log.success("   All done.")
