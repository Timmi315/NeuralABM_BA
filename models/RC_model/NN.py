import sys
from os.path import dirname as up

import coloredlogs
import h5py as h5
import numpy as np
import torch
from torchdiffeq import odeint
from dantro import logging
from dantro._import_tools import import_module_from_path
import random

from torchviz import make_dot

sys.path.append(up(up(__file__)))
sys.path.append(up(up(up(__file__))))

RC_model = import_module_from_path(mod_path=up(up(__file__)), mod_str="RC_model")
base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")

log = logging.getLogger(__name__)
coloredlogs.install(fmt="%(levelname)s %(message)s", level="INFO", logger=log)


class RC_model_NN:
    def __init__(
        self,
        *,
        rng: np.random.Generator,
        h5group: h5.Group,
        neural_net: base.NeuralNet,
        loss_function: dict,
        dt: float,
        true_parameters: dict = {},
        write_every: int = 1,
        write_start: int = 1,
        training_data: torch.Tensor,
        external_data: torch.Tensor,
        physical,
        batch_size: int,
        lookback: int,
        sample_size: int,
        mode: str,
        scaling_factors: dict = {},
        train_val_split = 0.7,
        **__,
    ):
        """Initialize the model instance with a previously constructed RNG and
        HDF5 group to write the output data to.

        :param rng (np.random.Generator): The shared RNG
        :param h5group (h5.Group): The output file group to write data to
        :param neural_net: The neural network
        :param loss_function (dict): the loss function to use
        :param to_learn: the list of parameter names to learn
        :param true_parameters: the dictionary of true parameters
        :param training_data: the time series of T_in data to calibrate
        :param external_data: the time series of the external data (T_out, Q_H, Q_O)
        :param write_every: write every iteration
        :param write_start: iteration at which to start writing
        :param batch_size: epoch batch size: instead of calculating the entire time series,
            only a subsample of length batch_size can be used. The likelihood is then
            scaled up accordingly.
        :param scaling_factors: factors by which the parameters are to be scaled
        """
        self._h5group = h5group
        self._rng = rng

        self.neural_net = neural_net
        self.neural_net.optimizer.zero_grad()
        self.loss_function = base.LOSS_FUNCTIONS[loss_function.get("name").lower()](
            loss_function.get("args", None), **loss_function.get("kwargs", {})
        )

        self.current_loss = torch.tensor(0.0)

        #self.to_learn = {key: idx for idx, key in enumerate(to_learn)}
        self.to_learn = {key: idx for idx, key in enumerate(physical.parameter_names)}
        self.true_parameters = {
            key: torch.tensor(val, dtype=torch.float)
            for key, val in true_parameters.items()
        }
        self.current_predictions = torch.zeros(len(self.to_learn))

        # Training data (time series of T_in) and external forcing (T_out, Q_H, Q_O)
        self.training_data = training_data
        self.train_set = slice(0, 1)
        if mode == "generalize":
            self.train_set = slice(0, int(training_data.shape[0]*train_val_split)+1)
            self.val_set = slice(self.train_set.stop, training_data.shape[0])
            log.info(f"\tNN running in generalize mode, train_set: {self.train_set}, val_set: {self.val_set}")
        
        #print("training_data", training_data)
        self.physical = physical

        # Time differential to use for the numerical solver
        self.dt = torch.tensor(dt).float()

        # Generate the batch ids
        self.batch_size = batch_size
        self.sample_size = sample_size
        
        self.mode = mode
        self.external_data = external_data

        #physical.external_data = external_data
        physical.dt = dt

        # Scaling factors to use for the parameters, if given
        self.scaling_factors = scaling_factors

        # --- Set up chunked dataset to store the state data in --------------------------------------------------------
        # Write the loss after every batch
        self._dset_loss = self._h5group.create_dataset(
            "loss",
            (0,),
            maxshape=(None,),
            chunks=True,
            compression=3,
        )
        self._dset_loss.attrs["dim_names"] = ["batch"]
        self._dset_loss.attrs["coords_mode__batch"] = "start_and_step"
        self._dset_loss.attrs["coords__batch"] = [write_start, write_every]

        # Write the computation time of every epoch
        self.dset_time = self._h5group.create_dataset(
            "computation_time",
            (0,),
            maxshape=(None,),
            chunks=True,
            compression=3,
        )
        self.dset_time.attrs["dim_names"] = ["epoch"]
        self.dset_time.attrs["coords_mode__epoch"] = "trivial"

        # Write the parameter predictions after every batch
        self.dset_parameters = self._h5group.create_dataset(
            "parameters",
            (0, len(self.to_learn.keys())),
            maxshape=(None, len(self.to_learn.keys())),
            chunks=True,
            compression=3,
        )
        self.dset_parameters.attrs["dim_names"] = ["batch", "parameter"]
        self.dset_parameters.attrs["coords_mode__batch"] = "start_and_step"
        self.dset_parameters.attrs["coords__batch"] = [write_start, write_every]
        self.dset_parameters.attrs["coords_mode__parameter"] = "values"
        self.dset_parameters.attrs["coords__parameter"] = physical.parameter_names


        self._time = 0
        self._write_every = write_every
        self._write_start = write_start

        self.lookback = lookback


    def shuffle_data(self, subset):
        data_to_shuffle = self.training_data
        if data_to_shuffle.shape[1] < self.sample_size or data_to_shuffle.shape[1] < self.lookback:
            print("Error: batch_size or lookback lie out of provided data range!")
            raise IndexError
        #only returns indices in form of (dataset, index:index+batch_size) to be used as torch slicers
        windows_per_series = int(self.batch_size)
        out = []
        for i in range(windows_per_series):
            for k in range(subset.start, subset.stop): #for every timeseries iff mode == "generalize" else only first one
                index = np.random.randint(data_to_shuffle.shape[1]-max(self.sample_size, self.lookback)+1)
                out.append((k, slice(index, index+max(self.sample_size, self.lookback), 1)))
        random.shuffle(out)
        return out

    def predict_and_simulate(self, data, batch_no, batch):
        # Make a prediction
        #print("batch train", self.training_data[batch][:self.lookback])
        #print("batch ext", self.external_data[batch][:self.lookback])
        newdata = torch.cat((data[batch][:self.lookback],
                                            self.external_data[batch][:self.lookback]),1)
        if self.mode == "single-input":
            nn_input = torch.flatten(data[batch][0])
            
        elif self.neural_net.type == "lstm":
            nn_input = newdata[:,:,0].unsqueeze(1)
        else: #predict or generalize mlp or optimizer
            nn_input = torch.flatten(newdata)


        predicted_parameters = self.neural_net(
            nn_input
        )


        # Get the parameters: resistance and capacity
        parameters = torch.stack([self.true_parameters[key]
                        if key in self.true_parameters.keys()
                        else self.scaling_factors.get(key , 1.0)*predicted_parameters[self.to_learn[key]]
                        for key in self.physical.parameter_names])

        self.physical.set_params(parameters)
        self.physical.reset()
               
        # Get current initial condition and make traceable
        current_densities = data[batch][0].clone()
        current_densities.requires_grad_(True)

        loss = torch.tensor(0.0, requires_grad=True)
        time = torch.arange(1, self.sample_size + 1, dtype = torch.float32)
        #print("time", time)
        self.physical.external_data = self.external_data[batch]
        #print(f"time.shape: {time.shape}")
        #print(f"y0.shape: {current_densities.shape}")
                
        trajectory = odeint(self.physical,
                                current_densities,
                                time,
                                method = 'euler',
                                options={'step_size': 1.0}
                                )
        #print(any(trajectory == float('nan')))
        #print("trajectory", trajectory)
        #print(trajectory.requires_grad)
        #print(trajectory.shape)
        #print(self.training_data[batch_idx + 1: self.batches[batch_no+1] + 1].shape)
        loss = self.loss_function(trajectory, data[batch[0], batch[1].start:batch[1].start+self.sample_size]) / (batch[1].stop- batch[1].start) #normalize loss over batch
        return loss, parameters


    def epoch(self):
        """
        An epoch is a pass over the entire dataset. The dataset is processed in batches, where B < L is the batch
        number. After each batch, the parameters of the neural network are updated. For example, if L = 100 and
        B = 50, two passes are made over the dataset -- one over the first 50 steps, and one
        over the second 50. The entire time series is processed, even if L is not divisible into equal segments of
        length B. For instance, is B is 30, the time series is processed in 3 steps of 30 and one of 10.

        """

        # Process the training data in batches

        #for batch_no, batch_idx in enumerate(self.batches[:-1]):
        for batch_no, batch in enumerate(self.shuffle_data(self.train_set)):

            loss, parameters = self.predict_and_simulate(self.training_data, batch_no, batch)
            
            #print(list(self.neural_net.named_parameters()))
            #dot = make_dot(loss, params = dict(self.neural_net.named_parameters()))
            #dot.render("graph", format = "png", directory = "C:/Users/Timo/Documents/SublimeProjects/NeuralABMBA")
            #print(f"Loss: {loss.item()}")
            #print(f"Grad check: {list(self.neural_net.parameters())[0].grad}")

            loss.backward()
            self.neural_net.optimizer.step()
            self.neural_net.optimizer.zero_grad()
            if not self.mode == "generalize":
                self.current_loss = loss.clone().detach().cpu().numpy().item()
                self.current_predictions = parameters.clone().detach().numpy()
                #self.current_predictions = predicted_parameters.clone().detach().cpu().numpy()

                self._time += 1
                self.write_data()
        if self.mode == "generalize":
            loss = 0
            for batch_no, batch in enumerate(self.shuffle_data(self.val_set)):
                l, params = self.predict_and_simulate(self.training_data, batch_no, batch)
                loss += l/(self.batch_size*(self.val_set.stop - self.val_set.start))
            self.current_loss = loss.clone().detach().cpu().numpy().item()
            self.current_parameters = params.clone().detach().numpy()
            self._time += 1
            self.write_data()


            
            

    def write_data(self):
        """Write the current state (loss and parameter predictions) into the state dataset.

        In the case of HDF5 data writing that is used here, this requires to
        extend the dataset size prior to writing; this way, the newly written
        data is always in the last row of the dataset.
        """
        if self._time >= self._write_start and (self._time % self._write_every == 0):
            self._dset_loss.resize(self._dset_loss.shape[0] + 1, axis=0)
            self._dset_loss[-1] = self.current_loss
            #self._dset_loss[-1] = 1
            self.dset_parameters.resize(self.dset_parameters.shape[0] + 1, axis=0)
            self.dset_parameters[-1, :] = [
                self.current_predictions[self.to_learn[p]] for p in self.to_learn.keys()
            ]


