# This experiment

import numpy
from mlpro.oa.systems.pool.doublependulum import *
from mlpro.sl.pool.afct.fnn.pytorch.mlp import PyTorchMLP
from mlpro.bf.systems.basics import DemoScenario, System
from mlpro.sl import *
import torch.nn as nn
import torch.optim as opt
import torch
from dataset import SASDataset
global cycle_id

# Defining Global variables for data collection
mean_loss_calculator = []
cycle_id = []
curr_cycle_id = 0
loss_collector = []
single_step_distance = []
continuous_distance = []

# Getting the datasets to obtain normalizers for MLP
# Paths Data Resource
train_path = os.curdir + os.sep + "data" + os.sep + 'training_data'
name_train_states = 'env_states.csv'
name_train_actions = 'agent_actions.csv'


# Obtaining Normalizer based on dataset
dp_instance = DoublePendulumSystemS4()
state_space, action_space = dp_instance.setup_spaces()


dataset = SASDataset(p_path=train_path,
                                    p_state_fname=name_train_states,
                                    p_action_fname=name_train_actions,
                                    p_state_space=state_space,
                                    p_action_space=action_space,
                                    # p_op_state_indexes=[0,2],
                                    p_normalize=True,
                                    p_batch_size=16,
                                    p_eval_split=0.3,
                                    p_shuffle=True,
                                    p_logging=Log.C_LOG_WE)



# Defining Custom Online Adapitve Simulation Scenario
class OnlineAdaptiveScenario(DemoScenario):

    def __init__(self,
                 p_reference_system: System,
                 p_oa_system: OASystem,
                 p_mode,
                 p_action_pattern: str = 'random',
                 p_action: list = None,
                 p_id=None,
                 p_cycle_limit=0,
                 p_auto_setup: bool = True,
                 p_visualize: bool = True,
                 p_logging=Log.C_LOG_ALL):

        self._oa_system = p_oa_system

        DemoScenario.__init__(self,
                              p_system=p_reference_system,
                              p_mode = p_mode,
                              p_action_pattern=p_action_pattern,
                              p_action = p_action,
                              p_id = p_id,
                              p_cycle_limit = p_cycle_limit,
                              p_auto_setup = p_auto_setup,
                              p_visualize = p_visualize,
                              p_logging = p_logging)


    ## -------------------------------------------------------------------------------------------------
    def setup(self):

        """Set's up the system spaces."""
        DemoScenario.setup(self)
        self._oa_system.setup_spaces()

    ## -------------------------------------------------------------------------------------------------
    def get_latency(self) -> timedelta:

        """
        Returns the latency of the system.
        """
        DemoScenario.get_latency(self)

    ## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed):

        """
        Resets the Scenario and the system. Sets up the action and state spaces of the system.

        Parameters
        ----------
        p_seed
            Seed for the purpose of reproducibility.
        """
        DemoScenario._reset(self,p_seed)
        self._oa_system.reset(p_seed=p_seed)
        self._oa_system.init_plot()

    ## -------------------------------------------------------------------------------------------------
    def _run_cycle(self):

        """
        Runs the custom scenario cycle, collect single step MSE loss for training and the single step prediction error for the Adaptive Function.
        """

        self.log(Log.C_LOG_TYPE_I, "Generating new action")

        action = self._get_next_action()

        prev_state = self._system.get_state()

        self._system.process_action(p_action=action)

        next_state = self._system.get_state()

        self._oa_system.adapt(p_state = prev_state, p_action = action, p_state_new = next_state)

        prev_oa_state = self._oa_system.get_state()

        self._oa_system._set_state(prev_state)

        self._oa_system.process_action(p_action=action)

        single_step_distance.append(ESpace.distance(next_state.get_related_set(),
                                                    next_state, self._oa_system.get_state()))

        self._oa_system._set_state(prev_oa_state)

        global curr_cycle_id
        curr_cycle_id = self.get_cycle_id()+2

        self._oa_system.process_action(p_action=action)

        broken = self._system.compute_broken(p_state=self._system.get_state())

        return False, broken, False, False

    ## -------------------------------------------------------------------------------------------------
    def update_plot(self, **p_kwargs):

        DemoScenario.update_plot(self,**p_kwargs)
        self._oa_system.update_plot(**p_kwargs)





# Defining reference System
reference_system = DoublePendulumSystemS4(p_logging=Log.C_LOG_NOTHING,
                                          p_visualize=False)





# Defining custom MLP implementation with preprocesisng capacities.
class CustomMLP(PyTorchMLP):

    ## -------------------------------------------------------------------------------------------------
    def _adapt_offline(self, p_dataset: dict) -> bool:
        """
        Adaptation mechanism for PyTorch based model for offline learning.

        Parameters
        ----------
        p_dataset : dict
            a dictionary that consists of a set of data, which are splitted to 2 keys such as input
            and output. The value of each key is a torch.Tensor of the sampled data.

        Returns
        ----------
            bool
        """

        self._sl_model.train()

        for input, target in p_dataset:

            try:
                input = torch.tensor(input.get_values(), dtype=torch.float)
                target = torch.tensor(target.get_values(), dtype=torch.float)
            except:
                pass

            torch.manual_seed(self._sampling_seed)
            outputs = self.forward(torch.squeeze(input))

            torch.manual_seed(self._sampling_seed)
            try:
                target = self._target_preproc(p_target=target)
            except:
                pass

            self._loss = self._calc_loss(outputs, torch.squeeze(target))

            self._optimize(self._loss)

            if isinstance(self._loss, torch.Tensor):
                self._loss = self._loss.item()

            self._sampling_seed += 1

        loss_collector.append(self._loss)
        cycle_id.append(curr_cycle_id)

        return True

    def _input_preproc(self, p_input:torch.Tensor) -> torch.Tensor:
        p_input = p_input.numpy()
        sin_th1 = (np.sin(np.radians(p_input[0])))
        cos_th1 = (np.cos(np.radians(p_input[0])))
        sin_th2 = (np.sin(np.radians(p_input[2])))
        cos_th2 = (np.cos(np.radians(p_input[2])))
        input = dataset._normalizer_feature_data.normalize(np.array([sin_th1, cos_th1, p_input[1], sin_th2, cos_th2, p_input[3], p_input[4]]))
        input = torch.Tensor(input)

        return input

    def _output_postproc(self, p_output:torch.Tensor) -> torch.Tensor:
        output = dataset._normalizer_label_data.denormalize(p_output.detach().numpy())
        th1 = np.degrees(np.arctan2(output[0], output[1]))
        th2 = np.degrees(np.arctan2(output[3], output[4]))
        output = torch.Tensor([th1, output[2], th2, output[5]])
        return output

    def _target_preproc(self, p_target:torch.Tensor):
        p_target = torch.squeeze(p_target)
        proc_target = torch.zeros((len(p_target),(len(p_target[0])+2)))
        for i,target in enumerate(p_target):
            target = torch.squeeze(target).numpy()
            sin_th1 = (np.sin(np.radians(target[0])))
            cos_th1 = (np.cos(np.radians(target[0])))
            sin_th2 = (np.sin(np.radians(target[2])))
            cos_th2 = (np.cos(np.radians(target[2])))
            target = dataset._normalizer_label_data.normalize(np.array([sin_th1, cos_th1, target[1], sin_th2, cos_th2, target[3]]))
            target = torch.Tensor(target)
            proc_target[i] = target
        return proc_target




# Setting the state space with new dimensions after pre-processing
state_space.add_dim(Dimension('sin th1'))
state_space.add_dim(Dimension('cos th1'))
state_space.add_dim(Dimension('sin th2'))
state_space.add_dim(Dimension('cos th2'))
dims = state_space.get_dim_ids()

state_space = state_space.spawn([dims[4],dims[5],dims[1],dims[6],dims[7],dims[3]])


# Defining the OAFctStrans
oafct_strans = OAFctSTrans(p_afct_cls=CustomMLP,
                              p_state_space=state_space,
                              p_action_space=action_space,
                              p_output_elem_cls=State,
                              p_threshold=0,
                              p_buffer_size=50,
                              p_num_hidden_layers=3,
                              p_activation_fct=nn.LeakyReLU(0.5),
                              p_output_activation_fct=nn.LeakyReLU(1),
                              p_optimizer=opt.Adam,
                              p_batch_size=16,
                              p_metrics=[MSEMetric(p_logging=Log.C_LOG_NOTHING),
                                         MetricAccuracy(p_threshold=0, p_logging=Log.C_LOG_NOTHING)],
                              p_learning_rate=0.0005,
                              p_hidden_size=256,
                              p_loss_fct=nn.MSELoss,
                              p_visualize=False,
                              p_logging=Log.C_LOG_WE)






# Resetting the state space of the OAFctStrans
state_space, action_space = reference_system.setup_spaces()
oafct_strans._afct_strans.get_afct()._output_space = state_space




# Defining Online Adaptive System
oa_System = DoublePendulumOA4(p_visualize=False,
                             p_fct_strans=oafct_strans,
                              m1 = 5,
                              m2 = 5,
                              p_logging=Log.C_LOG_NOTHING)



# 10. Instantiating the scenario
oa_sim_scenario = OnlineAdaptiveScenario(
                 p_reference_system=reference_system,
                 p_oa_system=oa_System,
                 p_mode = Mode.C_MODE_SIM,
                 p_cycle_limit=3000,
                 p_auto_setup = True,
                 p_visualize = False,
                 p_logging=Log.C_LOG_NOTHING)




# Running the scenario
oa_sim_scenario.run()





# Plotting the loss during onlinea adaptive training of OAFctStrans
plt.clf()
fig = plt.figure()
plt.plot(cycle_id, loss_collector, "blue", alpha = 0.25)
plt.plot(cycle_id, numpy.convolve(loss_collector, np.ones(50) / 50, "same"), "blue")
plt.grid()
plt.xlabel("Cycle")
plt.ylabel("Mean Squared Error")
plt.show()


plt.clf()
plt.figure()
plt.plot(single_step_distance, "blue", alpha = 0.25)
plt.plot(np.convolve(single_step_distance, np.ones(50) / 50, "same"), "blue")
plt.xlabel("Index")
plt.ylabel("Prediction Error")
plt.grid()
plt.show()