

from refactored.doublependulum import MyDoublePendulumOA4
from mlpro.rl import *
from mlpro.oa.streams.tasks import NormalizerMinMax
from refactored.boundarydetector import MyBoundaryDetector
from mlpro.rl.pool.policies.randomgenerator import RandomGenerator



cycle_limit         = 50
logging             = Log.C_LOG_ALL
visualize           = True
plotting            = True



# Defining further parameters
adaptivity = True
range = Range.C_RANGE_NONE






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OADPScenario(RLScenario):


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:



        # Creating the Online Adaptive Double Pendulum Environment
        oa_environment = MyDoublePendulumOA4(p_range_max=Range.C_RANGE_THREAD, p_name='', p_ada=adaptivity, p_visualize=visualize, p_max_torque=10)

        # Creating the Boundary Detector Task
        task_bd = MyBoundaryDetector(p_name='Boundary Detector', p_visualize=visualize, p_range_max=range)

        # Creating the Normalizer Task
        task_norm = NormalizerMinMax(p_name='Normalizer', p_visualize=visualize, p_range_max=range)

        # Adding the boundary detector task to the Reward Workflow
        oa_environment.add_task_reward(p_task=task_bd)

        # Adding the normalizer task to the reward workflow
        oa_environment.add_task_reward(p_task=task_norm, p_pred_tasks=[task_bd])

        # Registering the event handler to Normalizer
        task_bd.register_event_handler(p_event_id=task_bd.C_EVENT_ADAPTED, p_event_handler=task_norm.adapt_on_event)

        # Switching off visualization of uninteresting  objects
        oa_environment.switch_visualization(p_object=oa_environment.get_workflow_strans(), p_visualize=False)
        oa_environment.switch_visualization(p_object=oa_environment.get_workflow_success(), p_visualize=False)
        oa_environment.switch_visualization(p_object=oa_environment.get_workflow_broken(), p_visualize=False)

        # 2.1 Setup environment
        self._env   = oa_environment


        # 2.2 Setup and return random action agent
        policy_random = RandomGenerator(p_observation_space=self._env.get_state_space(),
                                        p_action_space=self._env.get_action_space(),
                                        p_buffer_size=1,
                                        p_ada=p_ada,
                                        p_visualize=p_visualize,
                                        p_logging=p_logging)

        return Agent(
            p_policy=policy_random,
            p_envmodel=None,
            p_name='Smith',
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging
        )




# 3 Create your scenario and run some cycles
myscenario  = OADPScenario( p_mode=Mode.C_MODE_SIM,
                            p_ada=adaptivity,
                            p_cycle_limit=cycle_limit,
                            p_visualize=visualize,
                            p_logging=logging )

myscenario.reset(p_seed=3)
for dim in myscenario.get_env().get_state_space().get_dims():
    dim._boundaries = []

myscenario.run()
input("Enter to stop")