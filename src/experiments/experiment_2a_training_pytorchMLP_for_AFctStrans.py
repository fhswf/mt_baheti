# 1. Importing necessary packages
import os
from pathlib import Path
from mlpro.rl.pool.envs.doublependulum import *
from mlpro.sl.pool.afct.fnn.pytorch.mlp import *
from mlpro.sl import *
from mlpro.sl.models_train import SLScenario
from mlpro.bf.datasets.basics import *
from dataset import SASDataset
import torch.optim as opt
import torch.nn as nn
import pathlib as path

# 2. Setting Path variables for training and offline dataset resources (CSV files in this case).
save_path = os.curdir + os.sep + "data"

# 2.1 Training Resource
train_path = str(path.Path.cwd().parent.parent) + os.sep + "data" + os.sep + 'training_data'
name_train_states = 'env_states.csv'
name_train_actions = 'agent_actions.csv'

# 2.2 Inference Resources
inference_path = os.curdir + os.sep + "data" + os.sep + 'inference_data'
name_infer_states = 'env_states.csv'
name_infer_actions = 'agent_actions.csv'

# 3. Getting the state and action space of the Double Pendulum Environment
dp = DoublePendulumS4()
state_space, action_space = dp.setup_spaces()

dataset = SASDataset(p_path=train_path,
                     p_state_fname=name_train_states,
                     p_action_fname=name_train_actions,
                     p_state_space=state_space,
                     p_action_space=action_space,
                     p_normalize=True,
                     p_batch_size=16,
                     p_eval_split=0.3,
                     p_shuffle=True,
                     p_logging=Log.C_LOG_WE)


# 4. Setting up Demo Scenario
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MLPSLScenario(SLScenario):
    C_NAME = 'DP'

    ## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:
        self._dataset = dataset

        return PyTorchMLP(p_input_space=self._dataset._feature_space,
                          p_output_space=self._dataset._label_space,
                          p_output_elem_cls=BatchElement,
                          p_num_hidden_layers=3,
                          p_activation_fct=nn.LeakyReLU(0.5),
                          p_output_activation_fct=nn.LeakyReLU(1),
                          p_optimizer=opt.Adam,
                          p_batch_size=16,
                          p_metrics=[MSEMetric(p_logging=Log.C_LOG_NOTHING),
                                     MetricAccuracy(p_threshold=20, p_logging=Log.C_LOG_NOTHING)],
                          p_learning_rate=0.0005,
                          p_hidden_size=256,
                          p_loss_fct=nn.MSELoss,
                          p_logging=Log.C_LOG_WE)


# 5. Preparing parameters for Demo and Unit Test modes.
if __name__ == "__main__":
    # 2.1 Parameters for demo mode
    cycle_limit = 1000000
    num_epochs = 50
    logging = Log.C_LOG_WE
    visualize = True
    save_path = str(Path.home())
    plotting = True
else:
    # 2.2 Parameters for internal unit test
    cycle_limit = 10000
    num_epochs = 2
    logging = Log.C_LOG_NOTHING
    visualize = False
    save_path = None
    plotting = False

# 6. Instantiating the Training Class
training = SLTraining(p_scenario_cls=MLPSLScenario,
                      p_cycle_limit=cycle_limit,
                      p_num_epoch=num_epochs,
                      p_logging=logging,
                      p_path=save_path,
                      p_eval_freq=1,
                      p_collect_mappings=False,
                      p_plot_epoch_scores=True)

# 7. Running the training
training.run()

# 8. Reloading the scenario from saved results of previous training
scenario = SLScenario.load(p_filename=training.get_scenario().get_filename(),
                           p_path=training.get_scenario()._get_path())

# 9. Getting the model from the Scenario
# Get the model
model = scenario.get_model()
# Switch off the adaptivity of the model
model.switch_adaptivity(False)


# 10. Setting up an Inference Scenario
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class InferenceScenario(SLScenario):
    C_NAME = 'Inference'

    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:
        self._dataset = SASDataset(p_path=inference_path,
                                   p_state_fname=name_infer_states,
                                   p_action_fname=name_infer_actions,
                                   p_state_space=state_space,
                                   p_action_space=action_space,
                                   # p_op_state_indexes=[0,2],
                                   p_batch_size=1,
                                   p_shuffle=False,
                                   p_normalize=True,
                                   p_logging=Log.C_LOG_NOTHING)

        self._dataset._normalizer_feature_data = dataset._normalizer_feature_data
        self._dataset._normalizer_label_data = dataset._normalizer_label_data

        return model


# 11. Instantiating the scenario
new_scenario = InferenceScenario(p_path=save_path,
                                 p_collect_mappings=True,
                                 p_cycle_limit=300,
                                 p_get_mapping_plots=True,
                                 p_save_plots=True)

# 12. Running the scenario
new_scenario.run()
