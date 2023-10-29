from mlpro.oa.streams.basics import *
from mlpro.oa.streams.tasks import BoundaryDetector
from typing import Union, Iterable, List


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyBoundaryDetector(BoundaryDetector):

    ## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_inst_new: List[Instance]):
        """
        Method to check if the new instances exceed the current boundaries of the Set.

        Parameters
        ----------
        p_inst_new:list
            List of new instance/s added to the workflow

        Returns
        -------
        adapted : bool
            Returns true if there is a change of boundaries, false otherwise.
        """

        adapted = False

        for inst in p_inst_new:
            if isinstance(inst, Instance):
                feature_data = inst.get_feature_data()
            else:
                feature_data = inst

            # Storing the related set for events
            self._related_set = feature_data.get_related_set()

            dim = feature_data.get_related_set().get_dims()

            if len(self._scaler) == 1:
                self._scaler = np.repeat(self._scaler, len(dim), axis=0)

            for i, value in enumerate(feature_data.get_values()):
                boundary = dim[i].get_boundaries()
                if len(boundary) == 0 or boundary is None:
                    boundary = [0, 0]
                    dim[i].set_boundaries(boundary)
                    adapted = True

                if value < boundary[0]:
                    dim[i].set_boundaries([value * self._scaler[i], boundary[1]])
                    adapted = True
                elif value > boundary[1]:
                    dim[i].set_boundaries([boundary[0], value * self._scaler[i]])
                    adapted = True

        return adapted


    ## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id: str, p_event_object: Event):
        """
        Event handler for Boundary Detector that adapts if the related event is raised.

        Parameters
        ----------
            p_event_id
                The event id related to the adaptation.
            p_event_obj:Event
                The event object related to the raised event.

        Returns
        -------
            bool
                Returns true if adapted, false otherwise.
        """

        adapted = False

        try:
            bd_new = self._scaler * p_event_object.get_raising_object().get_boundaries()
            self._related_set = p_event_object.get_data()["p_related_set"]
            dims = self._related_set.get_dims()

            for i, dim in enumerate(dims):
                bd_dim_current = dim.get_boundaries()
                bd_dim_new = bd_new[i]

                if (bd_dim_new[0] != bd_dim_current[0]) or (bd_dim_new[1] != bd_dim_current[1]):
                    dim.set_boundaries(bd_dim_new)
                    adapted = True
        except:
            raise ImplementationError("Event not raised by a window")

        return adapted


