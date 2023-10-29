from mt_baheti.src.refactored.boundarydetector import MyBoundaryDetector
from mlpro.sl import *
from mlpro.bf.streams.tasks import Rearranger
from mlpro.oa.streams.tasks import *






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyAdaptiveScenario(OAScenario):
    C_NAME = 'Demo'

    ## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging):
        # 1 Import a stream from OpenML
        mlpro = StreamProviderMLPro(p_logging=p_logging)
        stream = mlpro.get_stream(p_name=StreamMLProRnd10D.C_NAME,
                                  p_mode=p_mode,
                                  p_visualize=p_visualize,
                                  p_logging=p_logging)

        # 2 Set up a stream workflow based on a custom stream task

        # 2.1 Creation of a tasks
        features = stream.get_feature_space().get_dims()
        features_new = [('F', features[1:3])]

        task_rearranger = Rearranger(p_name='t1',
                                     p_range_max=Task.C_RANGE_THREAD,
                                     p_visualize=p_visualize,
                                     p_logging=p_logging,
                                     p_features_new=features_new)

        task_rearranger.C_PLOT_VALID_VIEWS =  [ PlotSettings.C_VIEW_ND ]

        task_bd = MyBoundaryDetector(p_name='Demo Boundary Detector',
                                   p_ada=p_ada,
                                   p_visualize=p_visualize,
                                   p_logging=p_logging)

        task_norm = NormalizerMinMax(p_name='Demo MinMax Normalizer',
                                     p_ada=p_ada,
                                     p_visualize=p_visualize,
                                     p_logging=p_logging)

        task_norm.C_PLOT_VALID_VIEWS =  [ PlotSettings.C_VIEW_ND ]
        # 2.2 Creation of a workflow
        workflow = OAWorkflow(p_name='wf1',
                              p_range_max=OAWorkflow.C_RANGE_NONE,  # StreamWorkflow.C_RANGE_THREAD,
                              p_ada=p_ada,
                              p_visualize=p_visualize,
                              p_logging=p_logging)

        # 2.3 Addition of the task to the workflow
        workflow.add_task(p_task=task_rearranger)
        workflow.add_task(p_task=task_bd, p_pred_tasks=[task_rearranger])
        workflow.add_task(p_task=task_norm, p_pred_tasks=[task_bd])

        # 3 Registering event handlers for normalizer on events raised by boundaries
        task_bd.register_event_handler(BoundaryDetector.C_EVENT_ADAPTED, task_norm.adapt_on_event)

        # 4 Return stream and workflow
        return stream, workflow


cycle_limit = 50
logging = Log.C_LOG_ALL
visualize = True

# 2 Instantiate the stream scenario
myscenario = MyAdaptiveScenario(p_mode=Mode.C_MODE_REAL,
                                p_cycle_limit=cycle_limit,
                                p_visualize=visualize,
                                p_logging=logging)

# 3 Reset and run own stream scenario
myscenario.reset()

myscenario.init_plot()
input('Press ENTER to start stream processing...')

myscenario.run()

input('Press ENTER to exit...')