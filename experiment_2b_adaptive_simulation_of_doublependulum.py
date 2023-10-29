from mlpro.sl.pool.afct.fnn.pytorch.mlp import *
import torch.nn as nn
import torch.optim as opt
from mlpro.sl.models_train import *
from mlpro.sl import *
from dataset import SASDataset
from mlpro.oa.systems.pool.doublependulum import *


# Defining global variables
error = []
progressive_error = []

# Data Resource
train_path = os.curdir + os.sep + "data" + os.sep + 'training_data'
name_train_states = 'env_states.csv'
name_train_actions = 'agent_actions.csv'




# Setting up Dataset to obtain the normalizers
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


# Defining customised MLP with preprocessinng and postprocssing functionalities
class CustomMLP(PyTorchMLP):

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

    # def _adapt_offline(self, p_dataset:dict) -> bool:


# Setting up adaptive state transition function
customAFctStrans = AFctSTrans(p_afct_cls=CustomMLP,
                              p_state_space=state_space,
                              p_action_space=action_space,
                              p_output_elem_cls=State,
                              p_threshold=0,
                              p_buffer_size=100,
                              p_num_hidden_layers=3,
                              p_activation_fct=nn.LeakyReLU(0.5),
                              p_output_activation_fct=nn.LeakyReLU(1),
                              p_optimizer=opt.Adam,
                              p_batch_size=200,
                              p_metrics=[MSEMetric(p_logging=Log.C_LOG_NOTHING),
                                         MetricAccuracy(p_threshold=10, p_logging=Log.C_LOG_NOTHING)],
                              p_learning_rate=0.0001,
                              p_hidden_size=256,
                              p_loss_fct=nn.MSELoss,
                              p_visualize=False,
                              p_logging=Log.C_LOG_WE)



# Getting the pre-trained model
from mlpro.sl.models_train import SLScenario
scenario = SLScenario.load(p_path= os.curdir + os.sep + "pre_trained_model" + os.sep + "mlp_model",
                           p_filename="InferenceScenario[a75c2578-1af4-44dd-abae-4e46efb34c77].pkl")
# Get the model
model = scenario.get_model()._sl_model
# Switch off the adaptivity of the model


# Setting up an adaptive DoublePendulumSystem
adaptiveDoublePendulum = DoublePendulumOA4(p_max_torque=10,
                                           p_fct_strans=customAFctStrans,
                                           p_visualize=False,
                                           p_logging=Log.C_LOG_ALL)



# 4. Defining Custom Online Adapitve Simulation Scenario
class ComparativeScenario(DemoScenario):

    def __init__(self,
                 p_reference_system: System,
                 p_adaptive_system: OASystem,
                 p_mode,
                 p_action_pattern: str = 'random',
                 p_action: list = None,
                 p_id=None,
                 p_cycle_limit=0,
                 p_auto_setup: bool = True,
                 p_visualize: bool = True,
                 p_logging=Log.C_LOG_ALL):

        self._adaptive_system = p_adaptive_system

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
        self._adaptive_system.setup_spaces()

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
        self._adaptive_system.reset(p_seed=p_seed)
        self._adaptive_system.init_plot()

    ## -------------------------------------------------------------------------------------------------
    def _run_cycle(self):

        """
        Runs the custom scenario cycle, for calculating both single step and progressive prediction error.
        """

        self.log(Log.C_LOG_TYPE_I, "Generating new action")

        action = self._get_next_action()

        prev_state = self._system.get_state()

        self._system.process_action(p_action=action)

        next_state = self._system.get_state()

        current_ada_state = self._adaptive_system.get_state()

        # Setting the state of the ASystem to reference system
        self._adaptive_system._set_state(prev_state)

        self._adaptive_system.process_action(p_action=action)

        # Calculating single step error
        predicted_state = self._adaptive_system.get_state()

        error.append(ESpace.distance(next_state.get_related_set(), next_state, predicted_state))

        # Setting the state of ASystem to previously predicted state
        self._adaptive_system._set_state(current_ada_state)

        self._adaptive_system.process_action(p_action=action)

        predicted_state = self._adaptive_system.get_state()

        # Calculating the progressive error
        progressive_error.append(ESpace.distance(next_state.get_related_set(), next_state, predicted_state))

        broken = self._system.compute_broken(p_state=self._system.get_state())

        return False, broken, False, False

    ## -------------------------------------------------------------------------------------------------
    def update_plot(self, **p_kwargs):

        DemoScenario.update_plot(self,**p_kwargs)
        self._adaptive_system.update_plot(**p_kwargs)


# Assigning the pre_trained model as the internal sl model for AFctStrans
adaptiveDoublePendulum.get_fct_strans()._afct._sl_model = model

scenario = ComparativeScenario(p_reference_system=DoublePendulumOA4(p_max_torque=10, p_visualize=False),
                               p_adaptive_system=adaptiveDoublePendulum,
                               p_cycle_limit=50,
                               p_mode=Mode.C_MODE_SIM)



scenario.run()



# Generating Plots
plt.figure()
plt.plot(error)
plt.xlabel("Cycle ID")
plt.ylabel("Prediction Error")
plt.grid()
plt.show()

plt.figure()
plt.plot(progressive_error)
plt.xlabel("Cycle ID")
plt.ylabel("Prediction Error")
plt.grid()
plt.show()


























