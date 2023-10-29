
from mlpro.rl.pool.envs.doublependulum import DoublePendulumOA4
import numpy as np
from mlpro.rl.models import State, Reward
from numpy import sin,cos

class MyDoublePendulumOA4(DoublePendulumOA4):
    ## ------------------------------------------------------------------------------------------------------
    def compute_reward_003(self, p_state_old: State = None, p_state_new: State = None):
        """
        Reward strategy with both new and old normalized state based on euclidean distance from the goal state,
        designed for the swinging of outer pole. Both angles and velocity and acceleration of the outer pole are
        considered for the reward computation.

        Parameters
        ----------
        p_state_old : State
            Normalized old state.
        p_state_new : State
            Normalized new state.

        Returns
        -------
        current_reward : Reward
            current calculated Reward values.
        """
        current_reward = Reward()
        state_new = p_state_new.get_values().copy()
        state_new[1] = 0
        try:
            state_new[4] = 0
        except:
            pass
        norm_state_new = State(self.get_state_space())
        norm_state_new.set_values(state_new)

        state_old = p_state_old.get_values().copy()
        try:
            state_old[1] = 0
        except:
            pass
        try:
            state_old[4] = 0
        except:
            norm_state_old = State(self.get_state_space())
        norm_state_old.set_values(state_old)
        goal_state = self._target_state

        d_old = abs(self.get_state_space().distance(goal_state, norm_state_old))
        d_new = abs(self.get_state_space().distance(goal_state, norm_state_new))
        d = d_old - d_new

        current_reward.set_overall_reward(d)

        return current_reward

    ## ------------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure, p_settings):
        """
        Custom method to initialize a 2D plot. If attribute p_settings.axes is not None the
        initialization shall be done there. Otherwise a new MatPlotLib Axes object shall be
        created in the given figure and stored in p_settings.axes.

        Parameters
        ----------
        p_figure : Matplotlib.figure.Figure
            Matplotlib figure object to host the subplot(s).
        p_settings : PlotSettings
            Object with further plot settings.
        """

        p_settings.axes = []

        p_settings.axes.append(p_figure.add_subplot(111, autoscale_on=False, xlim=(-self._L * 1.2, self._L * 1.2),
                                                    ylim=(-self._L * 1.2, self._L * 1.2)))
        p_settings.axes[0].set_aspect('equal')
        p_settings.axes[0].grid()
        p_settings.axes[0].set_title(self.C_NAME)


        self._line, = p_settings.axes[0].plot([], [], 'o-', lw=2)
        self._trace, = p_settings.axes[0].plot([], [], '.-', lw=1, ms=2)

    ## ------------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings, **p_kwargs):
        """
        This method updates the plot figure of each episode. When the figure is
        detected to be an embedded figure, this method will only set up the
        necessary data of the figure.
        """

        try:
            x1 = self._l1 * sin(self._y[:, 0])
            y1 = -self._l1 * cos(self._y[:, 0])

            x2 = self._l2 * sin(self._y[:, 2]) + x1
            y2 = -self._l2 * cos(self._y[:, 2]) + y1

            for i in range(len(self._y)):
                thisx = [0, x1[i], x2[i]]
                thisy = [0, y1[i], y2[i]]

                if i % self.C_ANI_FRAME == 0:
                    self._history_x.appendleft(thisx[2])
                    self._history_y.appendleft(thisy[2])
                    self._line.set_data(thisx, thisy)
                    self._trace.set_data(self._history_x, self._history_y)


        except:
            pass
