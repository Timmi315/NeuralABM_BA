import torch
import torch.nn as nn

class Physical(nn.Module):
    tame = 10e4
    def __init__(self):
        super().__init__()
    def reset(self):
        pass

    def tame_values(self, x):
        return (x*self.tame)/torch.sqrt(x**2 + self.tame**2)

    def set_params(self, params):
        self.params = params
        #print("setting params")

    def forward(self, t, state):
        #print(f"fw input: ({t}, {state})")
        #print(state[0] + self.dt/self.params[0]*((self.external_data[int(t.item())][0]-state[0])/self.params[1] + self.external_data[int(t.item())][1] + self.external_data[int(t.item())][2]))
        out = self.tame_values(self.step(state, self.external_data[int(t.item())], self.params, int(self.dt)))
        #print(out)
        #print(out)
        #out = torch.reshape(out, (1,self.dynamic_variables))
        #print(f"fw output: {out-state}")
        return out #since odeint expects df/dt -

class RC(Physical):
    parameter_names = ["C", "R"]
    dynamic_variables = 1
    data_names = ["T_in", "T_out", "heatPower", "solarGains"]
    plot_args = {
        "T_in": {
            "offset": -273.15,
            "multi": 1,
            "kwargs": {}
        },
        "T_out":{
            "offset": -273.15,
            "multi": 1,
            "kwargs": {}
        },
        "heatPower [kW]": {
            "offset": 0,
            "multi": 0.001,
            "kwargs": {"alpha": 0.5}
        },
        "Q_solar [kW]": {
            "offset" : 0,
            "multi": 0.001,
            "kwargs": {"alpha": 0.5}
        }
    }

    def __init__(self):
            super().__init__()

    def initial_condition(self, cfg, wdata):
        return [cfg["initial_conditions"]["T_in"], wdata[0, 2] + 273.15, 0, 0]


    def step(self, densities, data, params, dt):
        #for generation:
        # densities: dynamic variables
        # data: [T_ambient, heatPower(T_in), Q_solar]
        # params: parameters defined in config
        #for simulation in NN:
        # densities: first dynamic variables elements of data (one point in time)
        # data: rest of data generated through step() (one point in time)
        # params: estimated or set parameters in shape of parameter_names

        return (dt/params[0]*((data[0]-densities[0])/params[1] + data[1] + data[2])).unsqueeze(0)
    


class TiTh(Physical):
    parameter_names = ["C1", "R1", "C2", "R2"]
    dynamic_variables = 2
    data_names = ["T_in", "T_heater", "T_out", "heatPower", "solarGains"]
    plot_args = {
        "T_in": {
            "offset": -273.15,
            "multi": 1,
            "kwargs": {}
        },
        "T_heater": {
            "offset": -273.15,
            "multi": 1,
            "kwargs": {"alpha": 0.5}
        },
        "T_out":{
            "offset": -273.15,
            "multi": 1,
            "kwargs": {}
        },
        "heatPower [kW]": {
            "offset": 0,
            "multi": 0.001,
            "kwargs": {"alpha": 0.5}
        },
        "Q_solar": {
            "offset" : 0,
            "multi": 0.001,
            "kwargs": {"alpha": 0.5}
        }
    }

    def __init__(self):
            super().__init__()

    def initial_condition(self, cfg, wdata):
        return [cfg["initial_conditions"]["T_in"], cfg["initial_conditions"]["T_in"], wdata[0, 2] + 273.15, 0, 0]


    def step(self, densities, data, params, dt):
        #for generation:
        # densities: dynamic variables
        # data: [T_ambient, heatPower(T_in), Q_solar]
        # params: parameters defined in config
        #for simulation in NN:
        # densities: first dynamic variables elements of data (one point in time)
        # data: rest of data generated through step() (one point in time)
        # params: estimated or set parameters in shape of parameter_names
        P_T_in2Heater = (densities[0] - densities[1])/params[3]
        dT_heater = dt/params[2] * (data[1] + P_T_in2Heater)
        dT_in = dt/params[0] * ((data[0] - densities[0])/params[1] + data[2] - P_T_in2Heater)
        return torch.stack([dT_in, dT_heater])

    class Hidden(RC):
        parameter_names = ["C1", "R1", "C2", "R2"]
        T_heater = None

        def __init__(self):
            super().__init__()

        def initial_condition(self, cfg, wdata):
            return [cfg["initial_conditions"]["T_in"], cfg["initial_conditions"]["T_in"], wdata[0, 2] + 273.15, 0, 0]

        def step(self, densities, data, params, dt):
            # densities[1] -> data[0]
            if self.T_heater is None:
                self.T_heater = densities[0]
            self.T_heater = self.T_heater + self.tame_values(dt/params[2] * (data[1] + (densities[0] - self.T_heater) / params[3]))
            return (dt/params[0] * ((self.T_heater - densities[0])/params[3] + (data[0] - densities[0])/params[1] + data[2])).unsqueeze(0)

        def reset(self):
            self.T_heater = None

class TiTe(Physical):
    '''
    implementation of a 2R2C circuit with a capacity C2 inside the wall, connected to the inside ia R1 and the outside via R2
    '''
    parameter_names = ["C1", "R1", "C2", "R2"]
    dynamic_variables = 2
    data_names = ["T_in", "T_env", "T_out", "heatPower", "solarGains"]
    plot_args = {
        "T_in": {
            "offset": -273.15,
            "multi": 1,
            "kwargs": {}
        },
        "T_envelope": {
            "offset": -273.15,
            "multi": 1,
            "kwargs": {"alpha": 0.5}
        },
        "T_out":{
            "offset": -273.15,
            "multi": 1,
            "kwargs": {}
        },
        "heatPower [kW]": {
            "offset": 0,
            "multi": 0.001,
            "kwargs": {"alpha": 0.5}
        },
        "Q_solar": {
            "offset" : 0,
            "multi": 0.001,
            "kwargs": {"alpha": 0.5}
        }
    }

    def __init__(self):
            super().__init__()

    def initial_condition(self, cfg, wdata):
        return [cfg["initial_conditions"]["T_in"],
                cfg["initial_conditions"]["T_in"] + 273.15,
                wdata[0, 2] + 273.15]
    def step(self, densities, data, params, dt):
        P_in2Wall = (densities[0]-densities[1])/params[1] #the power flowing from C1 to C2
        dT_env = (P_in2Wall + (data[0]-densities[1])/params[3])*dt/params[2]
        dT_in = (data[1] + data[2] - P_in2Wall)*dt/params[0]
        return torch.stack([dT_in, dT_env])

    class Hidden(RC):
        parameter_names = ["C1", "R1", "C2", "R2"]
        T_envelope = None

        def initial_condition(self, cfg, wdata):
            return [cfg["initial_conditions"]["T_in"],
                (cfg["initial_conditions"]["T_in"] + wdata[0, 2] + 273.15)/2,
                wdata[0, 2] + 273.15]

        def step(self, densities, data, params, dt):
            if self.T_envelope is None:
                self.T_envelope = densities[0]
            P_in2Wall = (densities[0]-self.T_envelope)/params[1] #the power flowing from C1 to C2
            dT_env = (P_in2Wall + (data[0]-self.T_envelope)/params[3])*dt/params[2]
            self.T_envelope = self.T_envelope+self.tame_values(dT_env)
            dT_in = (data[1] + data[2] - P_in2Wall)*dt/params[0]
            return (dT_in).unsqueeze(0)

        def reset(self):
            self.T_envelope = None

class TiThTe(Physical):

    parameter_names = ["C1", "R1", "C2", "R2", "C3", "R3"]
    dynamic_variables = 3
    data_names = ["T_in", "T_heater", "T_env", "T_out", "heatPower", "solarGains"]
    plot_args = {
        "T_in": {
            "offset": -273.15,
            "multi": 1,
            "kwargs": {}
        },
        "T_heater": {
            "offset": -273.15,
            "multi": 1,
            "kwargs": {"alpha": 0.5}
        },
        "T_envelope": {
            "offset": -273.15,
            "multi": 1,
            "kwargs": {"alpha": 0.5}
        },
        "T_out":{
            "offset": -273.15,
            "multi": 1,
            "kwargs": {}
        },
        "heatPower [kW]": {
            "offset": 0,
            "multi": 0.001,
            "kwargs": {"alpha": 0.5}
        },
        "Q_solar": {
            "offset" : 0,
            "multi": 0.001,
            "kwargs": {"alpha": 0.5}
        }
    }

    def __init__(self):
            super().__init__()

    def initial_condition(self, cfg, wdata):
        return [cfg["initial_conditions"]["T_in"],
                cfg["initial_conditions"]["T_in"],
                cfg["initial_conditions"]["T_in"] + 273.15,
                wdata[0, 2] + 273.15]
    def step(self, densities, data, params, dt):
        P_in2Wall = (densities[0]-densities[2])/params[1] #the power flowing from ´the inside air to the envelope
        P_in2Heater = (densities[0]-densities[1])/params[3] # power flowing from the inside air into the radiator

        dT_heater = dt/params[2] * (data[1] + P_T_in2Heater)

        dT_env = (P_in2Wall + (data[0]-densities[2])/params[5])*dt/params[4]

        dT_in = (data[2] - P_in2Wall - P_in2Heater)*dt/params[0]
        return torch.stack([dT_in, dT_heater, dT_env])


    class Hidden(RC):
        parameter_names = ["C1", "R1", "C2", "R2", "C3", "R3"]
        T_heater = None
        T_envelope = None

        def __init__(self):
            super().__init__()

        def initial_condition(self, cfg, wdata):
            return [cfg["initial_conditions"]["T_in"],
                    cfg["initial_conditions"]["T_in"],
                    cfg["initial_conditions"]["T_in"] + 273.15,
                    wdata[0, 2] + 273.15]

        def reset(self):
            self.T_heater = None
            self.T_envelope = None

        def step(self, densities, data, params, dt):
            if self.T_heater is None or self.T_envelope is None:
                self.T_heater = densities[0]
                self.T_envelope = densities[0]

            P_in2Wall = (densities[0]-self.T_envelope)/params[1] #the power flowing from ´the inside air to the envelope
            P_in2Heater = (densities[0]-self.T_heater)/params[3] # power flowing from the inside air into the radiator

            self.T_heater = self.T_heater + self.tame_values(dt/params[2] * (data[1] + P_in2Heater))

            self.T_envelope = self.T_envelope + self.tame_values((P_in2Wall + (data[0]-self.T_envelope)/params[5])*dt/params[4])

            dT_in = (data[2] - P_in2Wall - P_in2Heater)*dt/params[0]
            return (dT_in).unsqueeze(0)



class EffWin(RC):
    def __init__(self):
            super().__init__()
    # Effective Window area can also be fitted -> data[2] contains HGloHor instead of solar_gains -> needs to be multiplied with A_eff
    parameter_names = ["C", "R", "A_eff"]
    def step(self, densities, data, params, dt):
        return (dt/params[0]*((data[0]-densities[0])/params[1] + data[1] + data[2]*params[2])).unsqueeze(0)

    class TiTe(TiTe.Hidden):
        def __init__(self):
            super().__init__()
            self.parameter_names = super().parameter_names
            self.parameter_names.append("A_eff")

        def step(self, densities, data, params, dt):
            return super().step(densities, torch.cat((data[:-1], (data[-1]*params[-1]).unsqueeze(0))), params[:-1], dt)

    class TiTh(TiTh.Hidden):
        def __init__(self):
            super().__init__()
            self.parameter_names = super().parameter_names
            self.parameter_names.append("A_eff")

        def step(self, densities, data, params, dt):
            return super().step(densities, torch.cat((data[:-1], (data[-1]*params[-1]).unsqueeze(0))), params[:-1], dt)

    class TiThTe(TiThTe.Hidden):
        def __init__(self):
            super().__init__()
            self.parameter_names = super().parameter_names
            self.parameter_names.append("A_eff")

        def step(self, densities, data, params, dt):
            #print(f"T_in: {densities}")
            #print(f"params: {params}")
            return super().step(densities, torch.cat((data[:-1], (data[-1]*params[-1]).unsqueeze(0))), params[:-1], dt)


