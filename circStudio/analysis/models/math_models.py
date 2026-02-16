import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from ..sleep.sleep import Roenneberg, Oakley, Crespo, Scripps, Sadeh, Cole_Kripke


class Model:
    """
    Numerically integrates a system of ordinary differential equations (ODEs), whose evolution may
    depend on external inputs (e.g., light) and initial conditions. Time points and input data are
    either provided directly or extracted from a pandas.Series with a DatetimeIndex.

    Attributes
    ----------
        initial_conditions : numpy.ndarray
            Array of initial conditions for the model states.
        model_states : numpy.ndarray or None
            Array containing the model state trajectories after integration.
        data : pandas.Series, optional
            Input data series with a DatetimeIndex, where the index specifies the time points and
            the values represent the input variable (e.g., light intensity). Time and value arrays
            are extracted from this series.
        time : numpy.ndarray, optional
            Array of time points (in hours); must be monotonically increasing.
        inputs : numpy.ndarray, optional
            Array of input values (e.g., light intensity) corresponding to each time point.
    """

    def __init__(self, initial_conditions, data=None, time=None, inputs=None):
        self.initial_conditions = initial_conditions
        self.model_states = None
        self.data = data

        # Extract time from data index
        if time is None or inputs is None:
            if data is not None:
                self.time = np.asarray(
                    (data.index - data.index.min()).total_seconds() / 3600
                )
                self.inputs = np.asarray(data.values)
            else:
                raise ValueError(
                    "Must provide either light time series (data) or input and time."
                )
        else:
            self.time = time
            self.inputs = inputs

    def initialize_model_states(self):
        """
        Initializes the model states by numerically integrating the system equations.

        This method runs the model integration using the current initial conditions,
        input values, and time vector, and stores the result in `self.model_states`.
        """
        self.model_states = self.integrate()

    def integrate(self, light_vector=None, time_vector=None, initial_condition=None):
        """
        Numerically integrates a system of ordinary differential equations (ODEs).

        This method uses SciPy `odeint` function to simulate the model dynamics over time, given a set of
        initial conditions and external inputs (e.g., light intensity). By default, it uses the class attributes
        `self.inputs`, `self.time`, and `self.initial_condition`, but alternative arrays can be provided.

        The input and time vectors must be of the same length, and the system's equations must be defined
        by the `derivative`method (which should be implemented within subclass of the Model class).

        Parameters
        ----------
        light_vector (numpy.ndarray, optional):
            Array of input values (typically, light intensity).
        time_vector (numpy.ndarray, optional):
            Array of time points (typically, time in hours).
        initial_condition (numpy.ndarray, optional):
            Array of initial conditions.

        Returns
        -------
        numpy.ndarray:
            Simulated state trajectories over the specified time vector.
        """
        light_vector = light_vector if light_vector is not None else self.inputs
        time_vector = time_vector if time_vector is not None else self.time
        initial_condition = (
            initial_condition
            if initial_condition is not None
            else self.initial_conditions
        )

        # Define a function to include light input
        def system_with_light(state, t):
            # light_input = light_vector[min(int(t), len(light_vector) - 1)]
            # index = np.where(time_vector <= t)[0][-1]
            index = np.searchsorted(time_vector, t, side="right")
            index = min(index, len(time_vector) - 1)
            light_input = light_vector[index]
            return self.derivative(t, state, light_input)

        # Use odeint to integrate the system
        solution = odeint(system_with_light, initial_condition, time_vector)
        return solution

    def get_initial_conditions(
        self, loop_number, data=None, light_vector=None, time_vector=None
    ):
        """
        Attempts to equilibrate the model's initial conditions by repeated simulation given a light and time vector.

        This method iteratively integrates the model for a user-specified number of cycles to assess whether equilibrium
         is achieved, as indicated by the stabilization of the predicted DLMO (Dim Light Melatonin Onset).

        Parameters
        ----------
        loop_number (int):
            Number of consecutive simulation cycles performed to assess whether the state trajectory of the circadian
            system's state variables converges to equilibrium under the given input conditions.
        data (pandas.Series, optional):
            Input data series with a DatetimeIndex. The index represents time points and the values represent input
            data (e.g., light intensity). The time and input arrays are extracted from this series.
        light_vector (numpy.ndarray, optional):
            Array of input values (typically, light intensity).
        time_vector (numpy.ndarray, optional):
            Array of time points (typically, time in hours).

        Returns
        ----------
        numpy.ndarray:
            The final state of the model after attempting to reach equilibrium (either an equilibrated solution or last
            simulated state).
        """
        # Extract time from light index
        if time_vector is None or light_vector is None:
            if data is not None:
                time_vector = np.asarray(
                    (data.index - data.index.min()).total_seconds() / 3600
                )
                light_vector = np.asarray(data.values)
            else:
                raise ValueError("Must provide either light series or input and time.")
        else:
            time_vector = time_vector
            light_vector = light_vector

        # Initialize with default initial conditions
        initial_condition = self.initial_conditions
        # List to collect dlmos
        dlmos = []
        for i in range(loop_number):
            # Calculate the solution for the models
            solution = self.integrate(
                initial_condition=initial_condition,
                light_vector=light_vector,
                time_vector=time_vector,
            )
            # Set current model states the solution
            self.model_states = solution
            # Calculate dlmos from the solution
            dlmos.append(self.dlmos())
            # Update initial_condition for the next iteration
            initial_condition = solution[-1]
            # Exit early if entrainment is detected
            if i > 1 and np.isclose(dlmos[-1][-1], dlmos[-2][-1], atol=1e-3):
                # Print number of loops required for entrainment
                # print(f"The model entrained after {i} loops.")
                # Update model initial conditions to entrained state
                self.initial_conditions = solution[-1]

                # Return entrained model solution
                return solution[-1]
        # Non-entrainment message(free-running rhythm)
        # print(
        #   "The model did not entrain due to insufficient loops or unentrainable light schedule."
        # )
        # Return unentrained model solution
        self.initial_conditions = solution[-1]
        return solution[-1]

    def dlmos(self):
        """
        Calculates the predicted Dim Light Melatonin Onset (DLMO) time points using a fixed,
        pre-specified offset (default: seven hours).

        Returns
        -------
        numpy.ndarray:
            Array of time points (in hours) at which DLMO is expected to occur.
        """
        return self.cbt() - self.cbt_to_dlmo

    def plot(self, states=False, dlmo=False, cbtmin=False):
        # Create a new plotly figure
        fig = go.Figure()

        if states:
            # Calculate number of states available
            states = self.model_states.shape[1]

            if self.model_name == "Forger" or self.model_name == "Jewett":
                labels = ["x", "xc", "Light Drive"]
            elif self.model_name == "HannaySP":
                labels = ["Amplitude", "Phase", "Light Drive"]
            else:
                labels = [
                    "Ventral Amplitude",
                    "Dorsal Amplitude",
                    "Ventral Phase",
                    "Dorsal Phase",
                    "Light Drive",
                ]
            # Iterate over states and plot them
            for i in range(states):
                fig.add_trace(
                    go.Scatter(
                        x=self.data.index.astype(str),
                        y=self.model_states[:, i],
                        name=f"{labels[i]}",
                    )
                )
                fig.update_layout(
                    title="Model States",
                    xaxis=dict(title="Time"),
                    yaxis=dict(title="Model States"),
                )
            return fig

        if dlmo:
            # Plot daily predicted DLMO
            fig.add_trace(
                go.Scatter(
                    x=pd.Series(self.data.index.date.astype(str)).unique(),
                    y=self.dlmos() % 24,
                    name="Predicted DLMO",
                )
            )
            fig.update_layout(
                title="Predicted DLMO",
                xaxis=dict(title="Day"),
                yaxis=dict(title="DLMO time"),
            )
            return fig

        if cbtmin:
            # Plot daily predicted DLMO
            fig.add_trace(
                go.Scatter(
                    x=pd.Series(self.data.index.date.astype(str)).unique(),
                    y=self.cbt() % 24,
                    name="Predicted CBTmin",
                )
            )
            fig.update_layout(
                title="Predicted CBTmin",
                xaxis=dict(title="Day"),
                yaxis=dict(title="CBTmin time"),
            )
            return fig


class Forger(Model):
    """
    Implements the mathematical model of human circadian rhythms developed by Forger, Jewett and Kronauer [1].
    The formalism includes a representation of the biochemical conversion of the light signal into a drive on
    the circadian pacemaker, which is modeled as a van der Pol oscillator. This cubic model is characterized
    by three state variables: x, xc and n. While n can be interpreted as the proportion of activated photoreceptors,
    at a given time, x and xc cannot directly be mapped to specific physiological mechanisms. Instead, x and xc are
    used to predict biologically meaningful quantities, such as the core body temperature minimum (CBTmin).

    Our implementation closely follows the approach of the `circadian`package by Arcascope [2]. However, we use the
    more powerful LSODA integrator (via SciPy's `odeint`) for numerical integration, enabling the integration of the
    system using more complex light trajectories.

    Attributes
    ----------
    taux : float
        Intrinsic period of the oscillator (hours).
    mu : float
        stiffness of the van der Pol oscillator.
    g : float
        Light sensitivity scaling parameter.
    alpha_0 : float
        Baseline forward rate constant governing the photon-driven activation
        (or depletion) of photoreceptors. This parameter scales the rate at which
        incident light converts available photoreceptors into an active or
        “used” state.
    beta : float
        Backward rate constant governing photoreceptor recovery or regeneration.
        This parameter controls the rate at which used photoreceptors return to
        the available state, representing dark adaptation or biochemical
        recovery processes.
    p : float
        Power-law exponent for light-dependent photoreceptor activation. This
        parameter determines how sensitively the forward rate constant α scales
        with light intensity.
    i0 : float
        Reference light intensity used to normalize the input light intensity vector I.
    k : float
        Coupling coefficient scaling the influence of the photic drive B on the
        circadian pacemaker. It modulates the sensitivity of the oscillator to light.
    cbt_to_dlmo : float
        Time offset (in hours) from CBTmin to DLMO.
    initial_conditions : numpy.ndarray
        State vector at the start of simulation (default: [-0.0843259, -1.09607546, 0.45584306]).
    model_states : numpy.ndarray
        Integrated state trajectories of the model.
    time : numpy.ndarray
        Array of time points for simulation.
    inputs : numpy.ndarray
        Array of input values (e.g., light intensity) over time.

    Methods
    -------
    derivative(t, state, light)
        Computes the derivatives of the state variables at a given time and light input.
    amplitude()
        Calculates the amplitude of the oscillator from integrated states.
    phase()
        Calculates the phase angle of the oscillator from integrated states.
    cbt()
        Identifies the timing of core body temperature minima from integrated states.

    References
    ----------
    [1] Forger DB, Jewett ME, Kronauer RE. A Simpler Model of the Human Circadian Pacemaker.
    Journal of Biological Rhythms. 1999;14(6):533-538. doi:10.1177/074873099129000867

    [2] Tavella, F., Hannay, K., & Walch, O. (2023). Arcascope/circadian: Refactoring of readers
    and metrics modules, Zenodo, v1.0.2. https://doi.org/10.5281/zenodo.8206871
    """

    def __init__(
        self,
        data=None,
        inputs=None,
        time=None,
        taux=24.2,
        mu=0.23,
        g=33.75,
        alpha_0=0.05,
        beta=0.0075,
        p=0.50,
        i0=9500.0,
        k=0.55,
        cbt_to_dlmo=7.0,
        initial_condition=None,
    ):
        if inputs is None or time is None:
            super().__init__(
                data=data,
                initial_conditions=np.array([-0.0843259, -1.09607546, 0.45584306]),
            )
        else:
            super().__init__(
                inputs=inputs,
                time=time,
                initial_conditions=np.array([-0.0843259, -1.09607546, 0.45584306]),
            )
        # Check for a scenario in which a initial condition is provided
        if initial_condition is not None:
            self.initial_conditions = initial_condition
        # self.initial_conditions = np.array([-0.0843259, -1.09607546, 0.45584306])
        # self.inputs = inputs
        # self.time = time
        self.model_name = self.__class__.__name__
        self.taux = taux
        self.mu = mu
        self.g = g
        self.alpha_0 = alpha_0
        self.beta = beta
        self.p = p
        self.i0 = i0
        self.k = k
        self.cbt_to_dlmo = cbt_to_dlmo
        # self.model_states = self.integrate()
        self.initialize_model_states()

    def derivative(self, t, state, light):
        """
        Computes the derivatives of the state variables at a given time and light input.

        Parameters
        ----------
        t : float
            Current simulation time (in hours).
        state : numpy.ndarray
            Current values of the state variables at time t.
        light : float
            Light intensity input (in lux) at time t.

        Returns
        -------
        numpy.ndarray
            Derivatives at time t.
        """
        x = state[0]
        xc = state[1]
        n = state[2]

        alpha = self.alpha_0 * pow((light / self.i0), self.p)
        Bhat = self.g * (1.0 - n) * alpha * (1 - 0.4 * x) * (1 - 0.4 * xc)
        mu_term = self.mu * (xc - 4.0 / 3.0 * pow(xc, 3.0))
        taux_term = pow(24.0 / (0.99669 * self.taux), 2.0) + self.k * Bhat

        dydt = np.zeros_like(state)
        dydt[0] = np.pi / 12.0 * (xc + Bhat)
        dydt[1] = np.pi / 12.0 * (mu_term - x * taux_term)
        dydt[2] = 60.0 * (alpha * (1.0 - n) - self.beta * n)

        return dydt

    def amplitude(self):
        """
        Amplitude of the oscillator computed from the integrated state trajectory.

        Returns
        -------
        numpy.ndarray
            Amplitude at each time point: sqrt(x^2 + xc^2).
        """
        # Integrate model and extract amplitude
        x = self.model_states[:, 0]
        xc = self.model_states[:, 1]
        return np.sqrt(x**2 + xc**2)

    def phase(self):
        """
        Returns the phase angle of the oscillator computed from the integrated state trajectory.

        Returns
        -------
        numpy.ndarray
            Phase angle (radians) at each time point: arctangent(xc/x).
        """
        # Integrate model and extract phase
        x = self.model_states[:, 0]
        # Multiplying xc makes the phase move clockwise
        xc = -1.0 * self.model_states[:, 1]
        # Observe that np.angle(x + complex(0,1) * xc) == np.atan2(xc,x)
        # The y (in this case, xc) appears first in the atan2 formula
        return np.angle(x + complex(0, 1) * xc)
        # return np.atan2(xc,x)

    def cbt(self):
        """
        Time points corresponding to the predicted core bod temperature minima (CBTmin), derived from the
        state variable x.

        Returns
        -------
        numpy.ndarray
            Array of time points (in hours) where minima of x occur, corresponding to the CBTmin.
        """
        # Calculate time step (dt) between consecutive time points
        dt = np.diff(self.time)[0]
        # Invert cos(x) to turn the minima into maxima (peaks)
        inverted_x = -1.0 * self.model_states[:, 0]
        # Identify the indices where the minima occur
        cbt_min_indices, _ = find_peaks(inverted_x, distance=np.ceil(13.0 / dt))
        # Use the previous indices to find the cbtmin times
        cbtmin_times = self.time[cbt_min_indices]
        # if you want to know in clock time, just do cbtmin_times % 24
        return cbtmin_times


class Jewett(Model):
    """
    Implements a refined version of the Forger, Jewett and Kronauer (FJK) model of human circadian rhythms,
    containing a nonlinearity of degree seven [1]. Compared to the FJK model implemented in the Forger subclass,
    the revised model recovers strength more slowly when the rhythm is very weak (low amplitude), but recovers
    faster once it is close to a stable rhythm (high amplitude).

    This model is characterized by three state variables: x, xc and n. While n can be interpreted as the proportion
    of activated photoreceptors, at a given time, x and xc cannot directly be mapped to specific physiological
    mechanisms. Instead, x and xc are used to predict biologically meaningful quantities, such as the core body
    temperature minimum (CBTmin).

    Our implementation closely follows the approach of the `circadian`package by Arcascope [2]. However, we use the
    more powerful LSODA integrator (via SciPy's `odeint`) for numerical integration, enabling the integration of the
    system using more complex light trajectories.

    Attributes
    ----------
    taux : float
        Intrinsic period of the oscillator (hours).
    mu : float
        Stiffness of the van der Pol oscillator.
    g : float
        Light sensitivity scaling parameter.
    alpha_0 : float
        Baseline forward rate constant governing the photon-driven activation
        (or depletion) of photoreceptors. This parameter scales the rate at which
        incident light converts available photoreceptors into an active or
        “used” state.
    beta : float
        Backward rate constant governing photoreceptor recovery or regeneration.
        This parameter controls the rate at which used photoreceptors return to
        the available state, representing dark adaptation or biochemical
        recovery processes.
    p : float
        Power-law exponent for light-dependent photoreceptor activation. This
        parameter determines how sensitively the forward rate constant α scales
        with light intensity.
    i0 : float
        Reference light intensity used to normalize the input light intensity vector I.
    k : float
        Coupling coefficient scaling the influence of the photic drive B on the
        circadian pacemaker. It modulates the sensitivity of the oscillator to light.
    q : float
        Coefficient governing an additional effect of light whereby, while light
        is present, it acts to shorten the intrinsic period of the pacemaker in
        addition to shifting its phase and amplitude.
    phi_ref : float
        Reference phase offset relating the timing of the model state minimum
        (x_min) to the timing of core body temperature minimum (CBT_min).
    cbt_to_dlmo : float
        Time offset (in hours) from CBTmin to DLMO.
    initial_conditions : numpy.ndarray
        State vector at the start of simulation (default: [-0.10097101, -1.21985662, 0.50529415]).
    model_states : numpy.ndarray
        Integrated state trajectories of the model.
    time : numpy.ndarray
        Array of time points for simulation.
    inputs : numpy.ndarray
        Array of input values (e.g., light intensity) over time.

    Methods
    -------
    derivative(t, state, light)
        Computes the derivatives of the state variables at a given time and light input.
    amplitude()
        Calculates the amplitude of the oscillator from integrated states.
    phase()
        Calculates the phase angle of the oscillator from integrated states.
    cbt()
        Identifies the timing of core body temperature minima from integrated states.

    References
    ----------
    [1] Jewett ME, Forger DB, Kronauer RE. Revised Limit Cycle Oscillator Model of Human Circadian
    Pacemaker. Journal of Biological Rhythms. 1999;14(6):493-500. https://doi.org/10.1177/074873049901400608

    [2] Tavella, F., Hannay, K., & Walch, O. (2023). Arcascope/circadian: Refactoring of readers
    and metrics modules, Zenodo, v1.0.2. https://doi.org/10.5281/zenodo.8206871
    """

    def __init__(
        self,
        data=None,
        inputs=None,
        time=None,
        taux=24.2,
        mu=0.13,
        g=19.875,
        beta=0.013,
        k=0.55,
        q=1.0 / 3.0,
        i0=9500,
        p=0.6,
        alpha_0=0.16,
        phi_ref=0.8,
        cbt_to_dlmo=7.0,
        initial_condition=None,
    ):
        if inputs is None or time is None:
            super().__init__(
                data=data,
                initial_conditions=np.array([-0.10097101, -1.21985662, 0.50529415]),
            )
        else:
            super().__init__(
                inputs=inputs,
                time=time,
                initial_conditions=np.array([-0.10097101, -1.21985662, 0.50529415]),
            )
        # Check for a scenario in which a initial condition is provided
        if initial_condition is not None:
            self.initial_conditions = initial_condition

        # self.initial_conditions= np.array([-0.10097101, -1.21985662, 0.50529415])
        # self.inputs = inputs
        # self.time = time
        self.model_name = self.__class__.__name__
        self.taux = taux
        self.mu = mu
        self.g = g
        self.beta = beta
        self.k = k
        self.q = q
        self.i0 = i0
        self.p = p
        self.alpha_0 = alpha_0
        self.phi_ref = phi_ref
        self.cbt_to_dlmo = cbt_to_dlmo
        # self.model_states = self.integrate()
        self.initialize_model_states()

    def derivative(self, t, state, light):
        """
        Computes the derivatives of the state variables at a given time and light input.

        Parameters
        ----------
        t : float
            Current simulation time (in hours).
        state : numpy.ndarray
            Current values of the state variables at time t.
        light : float
            Light intensity input (in lux) at time t.

        Returns
        -------
        numpy.ndarray
            Derivatives at time t.
        """
        x = state[0]
        xc = state[1]
        n = state[2]

        alpha = self.alpha_0 * (light / self.i0) ** self.p
        Bhat = self.g * alpha * (1 - n) * (1 - 0.4 * x) * (1 - 0.4 * xc)
        mu_term = self.mu * (1.0 / 3.0 * x + 4.0 / 3.0 * x**3 - 256.0 / 105.0 * x**7)
        taux_term = pow(24.0 / (0.99729 * self.taux), 2) + self.k * Bhat

        dydt = np.zeros_like(state)
        dydt[0] = np.pi / 12 * (xc + mu_term + Bhat)
        dydt[1] = np.pi / 12 * (self.q * Bhat * xc - x * taux_term)
        dydt[2] = 60.0 * (alpha * (1 - n) - self.beta * n)

        return dydt

    def amplitude(self):
        """
        Amplitude of the oscillator computed from the integrated state trajectory.

        Returns
        -------
        numpy.ndarray
            Amplitude at each time point: sqrt(x^2 + xc^2).
        """
        # Integrate model and extract amplitude
        x = self.model_states[:, 0]
        xc = self.model_states[:, 1]
        return np.sqrt(x**2 + xc**2)

    def phase(self):
        """
        Returns the phase angle of the oscillator computed from the integrated state trajectory.

        Returns
        -------
        numpy.ndarray
            Phase angle (radians) at each time point: arctangent(xc/x).
        """
        # Integrate model and extract phase
        x = self.model_states[:, 0]
        # Multiplying xc makes the phase move clockwise
        xc = -1.0 * self.model_states[:, 1]
        # Observe that np.angle(x + complex(0,1) * xc) == np.atan2(xc,x)
        # The y (in this case, xc) appears first in the atan2 formula
        return np.angle(x + complex(0, 1) * xc)
        # return np.atan2(xc,x)

    def cbt(self):
        """
        Time points corresponding to the predicted core bod temperature minima (CBTmin), derived from the
        state variable x.

        Returns
        -------
        numpy.ndarray
            Array of time points (in hours) where minima of x occur, corresponding to the CBTmin.
        """
        # Calculate time step (dt) between consecutive time points
        dt = np.diff(self.time)[0]
        # Invert cos(x) to turn the minima into maxima (peaks)
        inverted_x = -1.0 * self.model_states[:, 0]
        # Identify the indices where the minima occur
        cbt_min_indices, _ = find_peaks(inverted_x, distance=np.ceil(13.0 / dt))
        # Use the previous indices to find the cbtmin times
        cbtmin_times = self.time[cbt_min_indices]
        # if you want to know in clock time, just do cbtmin_times % 24
        return cbtmin_times


class HannaySP(Model):
    """
    Implements the Hannay Single-Population (SP) model of human circadian rhythms [1]. It describes a population of
    weakly-coupled oscillators using a formalism with three state variables: R (collective amplitude), Psi (collective
    phase), and n (proportion of light receptors used). In contrast to the FJK formalism, all three state variables are
    directly biologically interpretable. The model is derived from the mathematical description of the rhythm within
    individual cells in the suprachiasmatic nucleus (SCN) of the hypothalamus, from which a coherent behavior emerges.

    Our implementation closely follows the approach of the `circadian`package by Arcascope [2]. However, we use the
    more powerful LSODA integrator (via SciPy's `odeint`) for numerical integration, enabling the integration of the
    system using more complex light trajectories.

    Attributes
    ----------
    tau : float
        Intrinsic period of the oscillator (hours).
    k : float
        Average coupling strength between clock neurons.
    gamma : float
        Width (dispersion) of the intrinsic frequency distribution across the
        population of clock neurons, reflecting variability between neurons.
    beta : float
        Backward rate constant governing photoreceptor recovery or regeneration.
        This parameter controls the rate at which used photoreceptors return to
        the available state, representing dark adaptation or biochemical
        recovery processes.
    a1 : float
        Coefficient that scales the first harmonic component of the single-cell phase-response curve
    a2 : float
        Coefficient that scales the first harmonic component of the single-cell phase-response curve
    betal1 : float
        Phase offset for first harmonic component of the single-cell phase-response curve (radians).
    betal2 : float
        Phase offset for second harmonic component of the single-cell phase-response curve (radians).
    sigma : float
        Factor scaling the effect of light input on the circadian pacemaker (B(t)).
    g : float
        Light sensitivity scaling parameter.
    alpha_0 : float
        Baseline forward rate constant governing the photon-driven activation
        (or depletion) of photoreceptors. This parameter scales the rate at which
        incident light converts available photoreceptors into an active or
        “used” state.
    delta : float
        Backward rate constant governing photoreceptor recovery or regeneration.
        This parameter controls the rate at which used photoreceptors return to
        the available state, representing dark adaptation or biochemical
        recovery processes.
    p : float
        Power-law exponent for light-dependent photoreceptor activation. This
        parameter determines how sensitively the forward rate constant α scales
        with light intensity.
    i0 : float
        Reference light intensity used to normalize the input light intensity vector I.
    cbt_to_dlmo : float
        Time offset (in hours) from CBTmin to DLMO.
    initial_conditions : numpy.ndarray
        State vector at the start of simulation (default: [0.82041911, 1.71383697, 0.52318122]).
    model_states : numpy.ndarray
        Integrated state trajectories of the model.
    time : numpy.ndarray
        Array of time points for simulation.
    inputs : numpy.ndarray
        Array of input values (e.g., light intensity) over time.

    Methods
    -------
    derivative(t, state, light)
        Computes the derivatives of the state variables at a given time and light input.
    amplitude()
        Returns the collective rhythm amplitude (R) over time.
    phase()
        Returns the collective phase (Psi) as a wrapped angle over time.
    cbt()
        Identifies the timing of core body temperature minima from the phase trajectory.

    References
    ----------
    [1] Hannay KM, Booth V, Forger DB. Macroscopic Models for Human Circadian Rhythms.
    Journal of Biological Rhythms. 2019;34(6):658-671. https://doi.org/10.1177/0748730419878298

    [2] Tavella, F., Hannay, K., & Walch, O. (2023). Arcascope/circadian: Refactoring of readers
    and metrics modules, Zenodo, v1.0.2. https://doi.org/10.5281/zenodo.8206871
    """

    def __init__(
        self,
        data=None,
        inputs=None,
        time=None,
        tau=23.84,
        k=0.06358,
        gamma=0.024,
        beta=-0.09318,
        a1=0.3855,
        a2=0.1977,
        betal1=-0.0026,
        betal2=-0.957756,
        sigma=0.0400692,
        g=33.75,
        alpha_0=0.05,
        delta=0.0075,
        p=1.5,
        i0=9325.0,
        cbt_to_dlmo=7.0,
        initial_condition=None,
    ):
        if inputs is None or time is None:
            super().__init__(
                data=data,
                initial_conditions=np.array([0.82041911, 1.71383697, 0.52318122]),
            )
        else:
            super().__init__(
                inputs=inputs,
                time=time,
                initial_conditions=np.array([0.82041911, 1.71383697, 0.52318122]),
            )
        # Check for a scenario in which a initial condition is provided
        if initial_condition is not None:
            self.initial_conditions = initial_condition
        # self.initial_conditions = np.array([0.82041911, 1.71383697, 0.52318122])
        # self.inputs = inputs
        # self.time = time
        self.model_name = self.__class__.__name__
        self.tau = tau
        self.k = k
        self.gamma = gamma
        self.beta = beta
        self.a1 = a1
        self.a2 = a2
        self.betal1 = betal1
        self.betal2 = betal2
        self.sigma = sigma
        self.g = g
        self.alpha_0 = alpha_0
        self.delta = delta
        self.p = p
        self.i0 = i0
        self.cbt_to_dlmo = cbt_to_dlmo
        # self.model_states = self.integrate()
        self.initialize_model_states()

    def derivative(self, t, state, light):
        """
        Computes the derivatives of the state variables at a given time and light input.

        Parameters
        ----------
        t : float
            Current simulation time (in hours).
        state : numpy.ndarray
            Current values of the state variables at time t.
        light : float
            Light intensity input (in lux) at time t.

        Returns
        -------
        numpy.ndarray
            Derivatives at time t.
        """
        R = state[0]
        Psi = state[1]
        n = state[2]

        alpha = self.alpha_0 * pow(light, self.p) / (pow(light, self.p) + self.i0)

        Bhat = self.g * (1.0 - n) * alpha
        A1_term_amp = (
            self.a1 * 0.5 * Bhat * (1.0 - pow(R, 4.0)) * np.cos(Psi + self.betal1)
        )
        A2_term_amp = (
            self.a2
            * 0.5
            * Bhat
            * R
            * (1.0 - pow(R, 8.0))
            * np.cos(2.0 * Psi + self.betal2)
        )
        LightAmp = A1_term_amp + A2_term_amp
        A1_term_phase = (
            self.a1 * Bhat * 0.5 * (pow(R, 3.0) + 1.0 / R) * np.sin(Psi + self.betal1)
        )
        A2_term_phase = (
            self.a2 * Bhat * 0.5 * (1.0 + pow(R, 8.0)) * np.sin(2.0 * Psi + self.betal2)
        )
        LightPhase = self.sigma * Bhat - A1_term_phase - A2_term_phase

        dydt = np.zeros_like(state)
        dydt[0] = (
            -1.0 * self.gamma * R
            + self.k * np.cos(self.beta) / 2.0 * R * (1.0 - pow(R, 4.0))
            + LightAmp
        )
        dydt[1] = (
            2 * np.pi / self.tau
            + self.k / 2.0 * np.sin(self.beta) * (1 + pow(R, 4.0))
            + LightPhase
        )
        dydt[2] = 60.0 * (alpha * (1.0 - n) - self.delta * n)

        return dydt

    def amplitude(self):
        """
        Collective rhythm amplitude (R) of the oscillator population.

        Returns
        -------
        numpy.ndarray
            Amplitude (R) at each time point.
        """
        # Integrate model and extract collective rhythm amplitude (r)
        return self.model_states[:, 0]

    def phase(self):
        """
        Collective ventral phase (Psi) of the oscillator population, wrapped to [-pi,pi].

        Returns
        -------
        numpy.ndarray
            Phase angle (radians) at each time point.
        """
        # Integrate model and extract collective phase (phi)
        phi = self.model_states[:, 1]
        x = np.cos(phi)
        y = np.sin(phi)
        return np.angle(x + complex(0, 1) * y)

    def cbt(self):
        """
        Time points corresponding to the predicted core bod temperature minima (CBTmin), derived from the
        collective phase trajectory.

        Returns
        -------
        numpy.ndarray
            Array of time points (in hours) where minima of cos(Psi) occur, corresponding to the CBTmin.
        """
        # Calculate time step (dt) between consecutive time points
        dt = np.diff(self.time)[0]
        # Invert cos(x) to turn the minima into maxima (peaks)
        inverted_x = -np.cos(self.model_states[:, 1])
        # Identify the indices where the minima occur
        cbt_min_indices, _ = find_peaks(inverted_x, distance=np.ceil(13.0 / dt))
        # Use the previous indices to find the cbtmin times
        cbtmin_times = self.time[cbt_min_indices]
        # if you want to know in clock time, just do cbtmin_times % 24
        return cbtmin_times


class HannayTP(Model):
    """
    Implements the Hannay Two-Population (TP) model of human circadian rhythms [1]. The neuroanatomy of the
    suprachiasmatic nucleus (SCN) in the hypothalamus distinguishes two primary populations: the ventral cluster,
    which receives most light input, and the dorsal cluster. To better reflect this neurophysiological organization,
    Hannay et al. extended the Single Population (SP) model to explicitly represent both the ventral and dorsal
    groups, resulting in the TP model. Similar to the HannaySP, all five state variables are directly biologically
    interpretable. The model is also derived from the mathematical description of the rhythm within individual cells,
    from which a coherent behavior emerges.

    Our implementation closely follows the approach of the `circadian`package by Arcascope [2]. However, we use the
    more powerful LSODA integrator (via SciPy's `odeint`) for numerical integration, enabling the integration of the
    system using more complex light trajectories.

    Attributes
    ----------
    tauv : float
        Intrinsic period of the ventral oscillator (hours).
    taud : float
        Intrinsic period of the dorsal oscillator (hours).
    kvv : float
        Average coupling strength within the ventral oscillator population.
    kdd : float
        Average coupling strength within the dorsal oscillator population.
    kvd : float
        Average strength from ventral to dorsal population.
    kdv : float
        Average strength from dorsal to ventral population.
    gamma : float
        Width (dispersion) of the intrinsic frequency distribution across the
        population of clock neurons, reflecting variability between neurons.
    a1 : float
        Coefficient that scales the first harmonic component of the single-cell phase-response curve
    a2 : float
        Coefficient that scales the first harmonic component of the single-cell phase-response curve
    betal1 : float
        Phase offset for first harmonic component of the single-cell phase-response curve (radians).
    betal2 : float
        Phase offset for second harmonic component of the single-cell phase-response curve (radians).
    sigma : float
        Factor scaling the effect of light input on the circadian pacemaker (B(t)).
    g : float
        Light sensitivity scaling parameter.
    alpha_0 : float
        Baseline forward rate constant governing the photon-driven activation
        (or depletion) of photoreceptors. This parameter scales the rate at which
        incident light converts available photoreceptors into an active or
        “used” state.
    delta : float
        Backward rate constant governing photoreceptor recovery or regeneration.
        This parameter controls the rate at which used photoreceptors return to
        the available state, representing dark adaptation or biochemical
        recovery processes.
    p : float
        Power-law exponent for light-dependent photoreceptor activation. This
        parameter determines how sensitively the forward rate constant α scales
        with light intensity.
    i0 : float
        Reference light intensity used to normalize the input light intensity vector I.
    cbt_to_dlmo : float
        Time offset (in hours) from CBTmin to DLMO.
    initial_conditions : numpy.ndarray
        State vector at the start of simulation (default: [0.82423745, 0.82304996, 1.75233424, 1.863457, 0.52318122]).
    model_states : numpy.ndarray
        Integrated state trajectories of the model.
    time : numpy.ndarray
        Array of time points for simulation.
    inputs : numpy.ndarray
        Array of input values (e.g., light intensity) over time.

    Methods
    -------
    derivative(t, state, light)
        Computes the derivatives of the state variables at a given time and light input.
    amplitude()
        Returns the collective ventral rhythm amplitude (Rv) over time.
    phase()
        Returns the collective ventral phase (Psiv) as a wrapped angle over time.
    cbt()
        Identifies the timing of core body temperature minima based on the ventral phase trajectory.

    References
    ----------
    [1] Hannay KM, Booth V, Forger DB. Macroscopic Models for Human Circadian Rhythms.
    Journal of Biological Rhythms. 2019;34(6):658-671. https://doi.org/10.1177/0748730419878298

    [2] Tavella, F., Hannay, K., & Walch, O. (2023). Arcascope/circadian: Refactoring of readers
    and metrics modules, Zenodo, v1.0.2. https://doi.org/10.5281/zenodo.8206871
    """

    def __init__(
        self,
        data=None,
        inputs=None,
        time=None,
        tauv=24.25,
        taud=24.0,
        kvv=0.05,
        kdd=0.04,
        kvd=0.05,
        kdv=0.01,
        gamma=0.024,
        a1=0.440068,
        a2=0.159136,
        betal=0.06452,
        betal2=-1.38935,
        sigma=0.0477375,
        g=33.75,
        alpha_0=0.05,
        delta=0.0075,
        p=1.5,
        i0=9325.0,
        cbt_to_dlmo=7.0,
        initial_condition=None,
    ):
        if inputs is None or time is None:
            super().__init__(
                data=data,
                initial_conditions=np.array(
                    [0.82423745, 0.82304996, 1.75233424, 1.863457, 0.52318122]
                ),
            )
        else:
            super().__init__(
                inputs=inputs,
                time=time,
                initial_conditions=np.array(
                    [0.82423745, 0.82304996, 1.75233424, 1.863457, 0.52318122]
                ),
            )
        # Check for a scenario in which a initial condition is provided
        if initial_condition is not None:
            self.initial_conditions = initial_condition

        # self.initial_conditions = np.array([0.82423745, 0.82304996, 1.75233424, 1.863457, 0.52318122])
        # self.inputs = inputs
        # self.time = time
        self.model_name = self.__class__.__name__
        self.tauv = tauv
        self.taud = taud
        self.kvv = kvv
        self.kdd = kdd
        self.kvd = kvd
        self.kdv = kdv
        self.gamma = gamma
        self.a1 = a1
        self.a2 = a2
        self.betal = betal
        self.betal2 = betal2
        self.sigma = sigma
        self.g = g
        self.alpha_0 = alpha_0
        self.delta = delta
        self.p = p
        self.i0 = i0
        self.cbt_to_dlmo = cbt_to_dlmo
        # self.model_states = self.integrate()
        self.initialize_model_states()

    def derivative(self, t, state, light):
        """
        Computes the derivatives of the state variables at a given time and light input.

        Parameters
        ----------
        t : float
            Current simulation time (in hours).
        state : numpy.ndarray
            Current values of the state variables at time t.
        light : float
            Light intensity input (in lux) at time t.

        Returns
        -------
        numpy.ndarray
            Derivatives at time t.
        """
        Rv = state[0]
        Rd = state[1]
        Psiv = state[2]
        Psid = state[3]
        n = state[4]

        alpha = self.alpha_0 * pow(light, self.p) / (pow(light, self.p) + self.i0)
        Bhat = self.g * (1.0 - n) * alpha

        A1_term_amp = (
            self.a1 * 0.5 * Bhat * (1.0 - pow(Rv, 4.0)) * np.cos(Psiv + self.betal)
        )
        A2_term_amp = (
            self.a2
            * 0.5
            * Bhat
            * Rv
            * (1.0 - pow(Rv, 8.0))
            * np.cos(2.0 * Psiv + self.betal2)
        )
        LightAmp = A1_term_amp + A2_term_amp
        A1_term_phase = (
            self.a1 * Bhat * 0.5 * (pow(Rv, 3.0) + 1.0 / Rv) * np.sin(Psiv + self.betal)
        )
        A2_term_phase = (
            self.a2
            * Bhat
            * 0.5
            * (1.0 + pow(Rv, 8.0))
            * np.sin(2.0 * Psiv + self.betal2)
        )
        LightPhase = self.sigma * Bhat - A1_term_phase - A2_term_phase

        dydt = np.zeros_like(state)
        dydt[0] = (
            -self.gamma * Rv
            + self.kvv / 2.0 * Rv * (1 - pow(Rv, 4.0))
            + self.kdv / 2.0 * Rd * (1 - pow(Rv, 4.0)) * np.cos(Psid - Psiv)
            + LightAmp
        )
        dydt[1] = (
            -self.gamma * Rd
            + self.kdd / 2.0 * Rd * (1 - pow(Rd, 4.0))
            + self.kvd / 2.0 * Rv * (1.0 - pow(Rd, 4.0)) * np.cos(Psid - Psiv)
        )
        dydt[2] = (
            2.0 * np.pi / self.tauv
            + self.kdv / 2.0 * Rd * (pow(Rv, 3.0) + 1.0 / Rv) * np.sin(Psid - Psiv)
            + LightPhase
        )
        dydt[3] = 2.0 * np.pi / self.taud - self.kvd / 2.0 * Rv * (
            pow(Rd, 3.0) + 1.0 / Rd
        ) * np.sin(Psid - Psiv)
        dydt[4] = 60.0 * (alpha * (1.0 - n) - self.delta * n)

        return dydt

    def amplitude(self):
        """
        Collective ventral rhythm amplitude (Rv) of the oscillator population.

        Returns
        -------
        numpy.ndarray
            Amplitude Rv at each time point.

        """
        # Integrate model and extract collective rhythm amplitude (r)
        return self.model_states[:, 0]

    def phase(self):
        """
        Collective ventral phase (Psiv) of the oscillator population, wrapped to [-pi,pi].

        Returns
        -------
        numpy.ndarray
            Phase angle (radians) at each time point.
        """
        # Integrate model and extract collective phase (phi)
        phi = self.model_states[:, 2]
        x = np.cos(phi)
        y = np.sin(phi)
        return np.angle(x + complex(0, 1) * y)

    def cbt(self):
        """
        Time points corresponding to the predicted core bod temperature minima (CBTmin), derived from the
        ventral phase trajectory.

        Returns
        -------
        numpy.ndarray
            Array of time points (in hours) where minima of cos(Psiv) occur, corresponding to the CBTmin.
        """
        # Calculate time step (dt) between consecutive time points
        dt = np.diff(self.time)[0]
        # Invert cos(x) to turn the minima into maxima (peaks)
        inverted_x = -np.cos(self.model_states[:, 2])
        # Identify the indices where the minima occur
        cbt_min_indices, _ = find_peaks(inverted_x, distance=np.ceil(13.0 / dt))
        # Use the previous indices to find the cbtmin times
        cbtmin_times = self.time[cbt_min_indices]
        # if you want to know in clock time, just do cbtmin_times % 24
        return cbtmin_times


class ESRI:
    """
    Compute the Entrainment Signal Regularity Index (ESRI) for a given light schedule.

    The ESRI quantifies the strength and regularity of a light schedule by measuring how clustered
    the phases of a simulated circadian oscillator become after a fixed-duration exposure. A high
    ESRI indicates strong, regular entrainment (phases converge to a similar value), whereas a low
    ESRI indicates weak or irregular entrainment.

    The computation follows this procedure:
        1. Slide a window of length `window_size_days` across the input time series in steps of
        `esri_time_step_hours`.
        2. For each window start time, initialize a HannaySP model with a small amplitude and phase.
        3. Simulate the oscillator under the window's light input.
        4. Record the final amplitude as the ESRI value for that window.

    Parameters
    ----------
    time : array_like
        Time points (in hours) corresponding to the provided light intensities.
    inputs : array_like
        Light intensity values (e.g., lux) at each time point.
    window_size_days : float, optional
        Size of the sliding window in days over which ESRI is computed (default: 4.0).
    esri_time_step_hours : float, optional
        Step size in hours for sliding the window (default: 1.0).
    initial_amplitude : float, optional
        Initial amplitude for the HannaySP model; should be low to represent maximal phase dispersion
        (default: 0.1).
    midnight_phase : float, optional
        Phase (in radians) corresponding to circadian midnight (default: 1.65238233).

    Attributes
    ----------
    window_size : float
        Window length in days.
    esri_time_step : float
        Step size in hours for sliding the window.
    initial_amplitude : float
        Initial oscillator amplitude for each simulation.
    midnight_phase : float
        Phase offset for circadian midnight (radians).
    time_vector : ndarray
        Time points (hours) used for ESRI calculation.
    light_vector : ndarray
        Light intensity values corresponding to `time_vector`.
    raw_values : pandas.DataFrame
        DataFrame indexed by window start time (hours) with column `esri` for computed values.
    mean : float
        Mean ESRI value across all windows.
    std : float
        Standard deviation of ESRI values across all windows.
    """

    def __init__(
        self,
        data=None,
        time=None,
        inputs=None,
        window_size_days=4.0,
        esri_time_step_hours=1.0,
        initial_amplitude=0.1,
        midnight_phase=1.65238233,
    ):
        # Set parameter values for ESRI calculation
        self.window_size = window_size_days
        self.esri_time_step = esri_time_step_hours
        self.initial_amplitude = initial_amplitude
        self.midnight_phase = midnight_phase

        # Extract time and light vector
        if time is None or inputs is None:
            if data is not None:
                self.time_vector = np.asarray(
                    (data.index - data.index.min()).total_seconds() / 3600
                )
                self.light_vector = np.asarray(data.values)
            else:
                raise ValueError(
                    "Must provide either light time series (data) or input and time."
                )
        else:
            self.time_vector = time
            self.light_vector = inputs

        # Calculate ESRI array, mean and standard deviation
        self.raw_values = self.calculate()
        self.mean = self.raw_values["esri"].mean()
        self.std = self.raw_values["esri"].std()

    def calculate(self):
        """
        Calculates the ESRI time points and corresponding regularity values.

        Returns:
        --------
        pandas.DataFrame
            Dataframe indexed by window start time (hours) with column:

            esri : float
                Final oscillator amplitude (ESRI value) for each window start time.
        """
        # Determine the model's time step from the provided time vector
        model_time_step = np.diff(self.time_vector)[0]

        # Create an array of time points corresponding to the movement of the sliding window
        esri_time = np.arange(
            self.time_vector[0],
            self.time_vector[-1] - self.window_size * 24,
            self.esri_time_step,
        )
        # Array to store esri values for each time point in esri_time
        esri_values = np.zeros_like(esri_time)

        # Move the sliding window to calculate all ESRI values
        for i, t in enumerate(esri_time):
            # Initial phase in radians for the current posiition of the sliding window
            current_phase_init = self.midnight_phase + (np.mod(t, 24.0) * np.pi / 12)

            # Define initial condition based on initial amplitude, current_phase_init and n
            model_initial_condition = np.array(
                [self.initial_amplitude, current_phase_init, 0.0]
            )

            # Generate array with model time points to later compute the corresponding light intensity values by linear interpolation
            model_time_points = np.arange(
                t, t + self.window_size * 24.0, model_time_step
            )

            # Using linear interpolation, calculate light intensity value at the specified model time points, based on the time and light vector.
            linterp_light_vector = np.interp(
                model_time_points, self.time_vector, self.light_vector
            )

            # Calculate the trajetory using the linterp_light_vector and the model_time_points
            model = HannaySP(
                inputs=linterp_light_vector,
                time=model_time_points,
                initial_condition=model_initial_condition,
                k=0.0,
                gamma=0.0,
            )

            # model amplitude at the end of the simulation
            esri_values[i] = model.model_states[-1, 0]

        # Any negative values are replaced with NaN
        esri_values[esri_values < 0] = np.nan

        # Pack the time point and the corresponding esri value into a dataframe
        esri_df = pd.DataFrame({"time": esri_time, "esri": esri_values})

        # Set 'time' column as the index of the dataframe
        esri_df = esri_df.set_index("time")
        return esri_df


class ModelComparer:
    """
    Framework for mapping state variables between circadian rhythm models.

    The ModelComparer provides a framework for relating the state variables
    of the Forger model to the state variables of the HannaySP model. It enables
    a direct comparison between the two models and facilitates a more interpretable,
    physiologically grounded understanding of the Forger model states by expressing
    them as functions of the circadian phase predicted by the HannaySP model.

        Parameters
    ----------
    data : pandas.Series or pandas.DataFrame, optional
        Time-indexed light exposure data. If provided, the time vector is
        inferred from the index (in hours since the first sample), and the
        light input is taken from the values.
    inputs : array-like, optional
        Light input time series. Must be provided together with `time` if
        `data` is not supplied.
    time : array-like, optional
        Time vector (in hours). Must be provided together with `inputs` if
        `data` is not supplied.
    equilibrate : bool, default False
        If True, both models are equilibrated prior to simulation by repeatedly
        looping the input light profile to estimate stable initial conditions.
    loop_number : int, default 10
        Number of light-profile repetitions used during equilibration.
    a1, a2 : float, default 1.0
        Amplitude parameters for the trigonometric mapping of the Forger state
        variables.
    p1, p2 : float, default 1.0
        Phase scaling parameters for the trigonometric mapping.
    m1, m2 : float, default 0.0
        Offset parameters for the trigonometric mapping.

    Attributes
    ----------
    time_vector : ndarray
        Time vector (hours).
    light_vector : ndarray
        Light input time series.
    forger : Forger
        Instance of the Forger model.
    hannay : HannaySP
        Instance of the HannaySP model.
    x, xc : ndarray
        State variables of the Forger model.
    phase_vector : ndarray
        Collective phase of the HannaySP model.
    predicted_x, predicted_xc : ndarray or None
        Predicted Forger state variables obtained from the HannaySP phase.
    """

    def __init__(
        self,
        data=None,
        inputs=None,
        time=None,
        equilibrate=False,
        loop_number=10,
        a1=1.0,
        a2=1.0,
        p1=1.0,
        p2=1.0,
        m1=0.0,
        m2=0.0,
    ):
        # Translation params
        self.a1 = a1
        self.a2 = a2
        self.p1 = p1
        self.p2 = p2
        self.m1 = m1
        self.m2 = m2

        # Extract time from data index
        if time is None or inputs is None:
            if data is not None:
                self.time_vector = np.asarray(
                    (data.index - data.index.min()).total_seconds() / 3600
                )
                self.light_vector = np.asarray(data.values)
            else:
                raise ValueError(
                    "Must provide either light time series (data) or input and time."
                )
        else:
            self.time_vector = time
            self.light_vector = inputs

        # Instantiate forger model
        self.forger = Forger(inputs=self.light_vector, time=self.time_vector)
        if equilibrate:
            # First, calculate initial conditions based on the light and time vector
            ics = self.forger.get_initial_conditions(
                loop_number=loop_number,
                light_vector=self.light_vector,
                time_vector=self.time_vector,
            )

            # Second, integrate the equilibrated model and use last state as initial conditions
            new_ics = self.forger.integrate(
                light_vector=self.light_vector,
                time_vector=self.time_vector,
                initial_condition=ics,
            )[-1]

            # Third, use the calculated initial conditions to define current model states
            self.forger.model_states = self.forger.integrate(
                light_vector=self.light_vector,
                time_vector=self.time_vector,
                initial_condition=ics,
            )

        # Instantiate hannay model
        self.hannay = HannaySP(inputs=self.light_vector, time=self.time_vector)
        if equilibrate:
            # First, calculate initial conditions based on the light and time vector
            ics = self.hannay.get_initial_conditions(
                loop_number=50,
                light_vector=self.light_vector,
                time_vector=self.time_vector,
            )

            # Second, integrate the equilibrated model and use last state as initial conditions
            new_ics = self.hannay.integrate(
                light_vector=self.light_vector,
                time_vector=self.time_vector,
                initial_condition=ics,
            )[-1]

            # Third, use the calculated initial conditions to define current model states
            self.hannay.model_states = self.hannay.integrate(
                light_vector=self.light_vector,
                time_vector=self.time_vector,
                initial_condition=new_ics,
            )

        # Store state variables of the forger model
        self.x = self.forger.model_states[:, 0]
        self.xc = self.forger.model_states[:, 1]

        # Store the collective phase of Hannay's model
        self.phase_vector = self.hannay.model_states[:, 1]

        # Predicted x and xc attributes
        self.predicted_x = None
        self.predicted_xc = None

    def predict_forger(self, change_params=True):
        """
        Predict Forger state variables using the collective phase predicted
        by the HannaySP model.

        Expresses the Forger state variables (`x`, `xc`) as explicit functions
        of the collective phase predicted by the HannaySP model.

        Parameters
        ----------
        change_params: bool, default True
        If True, stores the predicted values for the `x` and `xc` state variables
        in the attributes `predicted_x` and `predicted_xc`.

        Returns
        -------
        predicted_x : ndarray
            Predicted values of the Forger model state variable `x`
        predicted_xc : ndarray
            Predicted values of the Forger model state variable `xc`
        """
        # Predict state variable x in FJK model (either Forger or Jewett)
        predicted_x = np.array(
            [self.a1 * np.cos(self.p1 * phase) + self.m1 for phase in self.phase_vector]
        )

        # Predict state variable xc in FJK model (either Forger or Jewett)
        predicted_xc = np.array(
            [
                -1.0 * self.a2 * np.sin(self.p2 * phase) + self.m2
                for phase in self.phase_vector
            ]
        )

        # Define the predicted_x and predicted_xc attributes
        if change_params:
            self.predicted_x = predicted_x
            self.predicted_xc = predicted_xc

        # Return predictions for x and xc as a tuple
        return predicted_x, predicted_xc

    def linearize_phase(self, change_params=False):
        """
        Linearize the phase of the Forger model state variables.

        Fits a first-degree polynomial to the phase trajectory of the HannaySP
        and optionally replaces the stored phase vector with its linear approximation.

        Parameters
        ----------
        change_params: bool, default False
            If True, replaces `phase_vector` with the linearized phase.

        Returns
        -------
        linear_phase : ndarray
            Linear approximation of the collective phase.
        """
        # Attempt to fit a straight line (a polynomial of degree 1) to the collective phase
        coefficients = np.polyfit(self.time_vector, self.phase_vector, 1)
        slope, intercept = coefficients
        if change_params == True:
            self.phase_vector = slope * self.time_vector + intercept
        return slope * self.time_vector + intercept

    def find_optimal_params(self, change_params=True):
        """
        Estimate the optimal mapping parameters between HannaySP and Forger states.

        Uses non-linear least squares regression to estimate amplitude, offset, and
        phase-scaling parameters for the mapping between the HannaySP phase and the
        Forger model state variables `x` and `xc`.

        Parameters
        ----------
        change_params : bool, default True
            If True, updates the mapping parameters stored in the object.

        Returns
        -------
        params : ndarray, shape (2, 3)
            Estimated parameters for `x` and `xc` mappings:
            [[a1, m1, p1],
            [a2, m2, p2]]
        """

        # Define a mapping function for the state variable x in the FJK model
        def map_x(phase, A1, M1, P1):
            return A1 * np.cos(P1 * phase) + M1

        # Define a mapping function for the state variable xc in the FJK model
        def map_xc(phase, A2, M2, P2):
            return -1.0 * A2 * np.sin(P2 * phase) + M2

        # Initial values for A1 and A2, M1 and M2, and P1 and P2
        initial_guess = [1.0, 0.0, 1.0]

        def calculate_params(map_function, initial_guess, vector_to_predict, phase):
            # Find optimal parameters using non-linear least squares regression
            optimal_params, covariance = curve_fit(
                map_function, phase, vector_to_predict, p0=initial_guess
            )
            return optimal_params

        # Calculate A1, M1, P1, A2, M2 and P2
        a1, m1, p1 = calculate_params(map_x, initial_guess, self.x, self.phase_vector)
        a2, m2, p2 = calculate_params(map_xc, initial_guess, self.xc, self.phase_vector)

        # Change the attributes of the object
        if change_params:
            self.a1 = a1
            self.a2 = a2
            self.m1 = m1
            self.m2 = m2
            self.p1 = p1
            self.p2 = p2

        # Return a 2 * 3 matrix with parameters
        return np.array([[a1, m1, p1], [a2, m2, p2]])

    def rmse(self):
        """
        Compute the root-mean-square error (RMSE) of the predicted Forger model states.

        Returns
        -------
        rmse_x : float
            RMSE between observed and predicted `x`.
        rmse_xc : float
            RMSE between observed and predicted `xc`.

        Notes
        -------
        Requires `predict_forger` to be called beforehand with
        `change_params=True`.
        """
        rmse_x = np.sqrt(np.mean((self.x - self.predicted_x) ** 2))
        rmse_xc = np.sqrt(np.mean((self.xc - self.predicted_xc) ** 2))
        return rmse_x, rmse_xc

    def cumulative_rmse(self):
        """
        Compute cumulative RMSE for the predicted Forger model states.

        Returns
        -------
        crmse_x : ndarray
            Cumulative RMSE between observed and predicted `x`.
        crmse_xc : ndarray
            Cumulative RMSE for state variable `xc`.

        Notes
        -------
        Useful for assessing how prediction error evolves over time.
        Requires predicted values to be available.
        """
        error_x = (self.x - self.predicted_x) ** 2
        error_xc = (self.xc - self.predicted_xc) ** 2
        crmse_x = np.sqrt(np.cumsum(error_x) / np.arange(1, len(self.x) + 1))
        crmse_xc = np.sqrt(np.cumsum(error_xc) / np.arange(1, len(self.xc) + 1))
        return crmse_x, crmse_xc

    def half_life_crmse(self):
        """
        Fit an exponential decay function to the cumulative RMSE curves and compute
        the half life times of the RMSE for both the `x` and `xc` values.

        The half-life represents the time required for the prediction error to decrease
        by 50%, and can be used as an index of how readily a given light schedule entrains
        the circadian pacemaker.

        Returns
        -------
        half_life_x : float
            Half-decay time for the `x` state variable.
        half_life_xc : float
                Half-decay time for the `xc` state variable.
        """
        # Compute the cumulative rmse for x and xc state variables
        crmse_x, crmse_xc = self.cumulative_rmse()

        # Define exponential decay function
        def exponential_decay(t, A, k, C):
            return A * np.exp(-k * t) + C

        # Obtain optimal parameters to fit exponential decay
        def optimal_params(crmse):
            return curve_fit(
                exponential_decay,  # function
                self.time_vector,  # xdata
                crmse,  # ydata
                p0=[
                    crmse[0] - crmse[-1],  # Initial guess for the amplitude A
                    0.01,  # k: slow decay guess
                    crmse[-1],  # C: asymptotic RMSE value
                ],  # Initial guess for the parameters
                maxfev=10000,  # Iterations for curve fitting
            )[0]

        # Unpack optimal params for `x` and `xc`
        a_x, k_x, c_x = optimal_params(crmse_x)
        a_xc, k_xc, c_xc = optimal_params(crmse_xc)

        # Calculate the half-life for state variables
        half_life_x = np.inf if k_x <= 0 else np.log(2) / k_x
        half_life_xc = np.inf if k_xc <= 0 else np.log(2) / k_xc

        # Return the half-life for `x` and `xc` prediction
        return half_life_x, half_life_xc


class Hilaire07(Model):
    """
    St. Hilaire et al. (2007) human circadian pacemaker model.

    This model extends earlier light-driven circadian oscillator models (e.g.,
    Kronauer / Forger formulations) by including a non-photic component representing
    behavioral modulation (sleep/wake state).

    Our implementation closely follows the approach of the `circadian`package by Arcascope [2]. However, we use the
    more powerful LSODA integrator (via SciPy's `odeint`) for numerical integration, enabling the integration of the
    system using more complex light trajectories.

    Model inputs
    ------------
    The system is driven by:
    - Light intensity (lux)
    - Sleep/rest stat (binary; 1=sleep/rest, 0=wake)

    Sleep state may be computed automatically from actigraphy data using
    one of the supported sleep scoring algorithms available in circStudio
    (e.g., Cole-Kripke, Roenneberg), or supplied directly by the user.

    Methods
    -------
    derivative(t, state, light)
        Differential equation system of the oscillator.
    amplitude()
        Calculates the amplitude of the oscillator from integrated states.
    phase()
        Calculates the phase angle of the oscillator from integrated states.
    cbt()
        Identifies the timing of core body temperature minima from integrated states.

    Parameters
    ----------
    data : pandas.Series, optional
        Time-indexed light data. Required if `inputs` and `time` are not explicitly provided.
    sleep_algo : str or pandas.Series, optional
        Sleep scoring algorithm to use (default: "Roenneberg").
        Alternatively, a precomputed binary sleep series may be supplied.
    inputs : array-like, optional
        Light input.
    time : array-like, optional
        Time input.
    taux : float
        Intrinsic circadian period (hours).
    g, k, mu, beta, q, rho, i0, p, a0 : float
        Model parameters governing light response, non-photic drive,
        and oscillator dynamics.
    phi_xcx : float
        Polar phase angle (radians) between the state variables x and x_c.
    phi_ref : float
        Fixed phase offset (radians) used to convert internal oscillator phase
        to clock time.
    cbt_to_dlmo : float
        Phase relationship between CBT minimum and DLMO (hours).
    initial_condition : array-like, optional
        Initial state vector [-0.0480751, -1.22504441, 0.51854818].

    References
    ----------
    [1] St. Hilaire, M. A., Klerman, E. B., Khalsa, S. B. S., Wright, K. P., Czeisler, C. A., & Kronauer, R. E. (2007).
     Addition of a non-photic component to a light-based mathematical model of the human circadian pacemaker.
     Journal of Theoretical Biology, 247(4), 583–599. https://doi.org/10.1016/j.jtbi.2007.04.001

    [2] Tavella, F., Hannay, K., & Walch, O. (2023). Arcascope/circadian: Refactoring of readers
    and metrics modules, Zenodo, v1.0.2. https://doi.org/10.5281/zenodo.8206871
    """

    def __init__(
        self,
        data=None,
        sleep_algo='Roenneberg',
        inputs=None,
        time=None,
        taux=24.2,
        g=37.0,
        k=0.55,
        mu=0.13,
        beta=0.007,
        q=1.0 / 3.0,
        rho=0.032,
        i0=9500.0,
        p=0.5,
        a0=0.1,
        phi_xcx=-2.98,
        phi_ref=0.97,
        cbt_to_dlmo=7.0,
        initial_condition=None,
    ):
        if inputs is None or time is None:
            super().__init__(
                data=data,
                initial_conditions=np.array([-0.0480751, -1.22504441, 0.51854818]),
            )
        else:
            super().__init__(
                inputs=inputs,
                time=time,
                initial_conditions=np.array([-0.0480751, -1.22504441, 0.51854818]),
            )
        # Check for a scenario in which a initial condition is provided
        if initial_condition is not None:
            self.initial_conditions = initial_condition
        self.model_name = self.__class__.__name__
        self.taux = taux
        self.g = g
        self.k = k
        self.mu = mu
        self.beta = beta
        self.q = q
        self.rho = rho
        self.i0 = i0
        self.p = p
        self.a0 = a0
        self.phi_xcx = phi_xcx
        self.phi_ref = phi_ref
        self.cbt_to_dlmo = cbt_to_dlmo
        self.initialize_model_states()
        self.sleep = self.sleep_vector(algo=sleep_algo)

    def sleep_vector(self, algo='Roenneberg', **kwargs):
        """
        Return a binary sleep/rest series align to self.data.

        This method either:
            * Runs one of circStudio's supported sleep-detection algorithms
            on self.data or
            * Returns a user-supplied sleep series directly.

        The output is always a pandas Series indexed by the same DatetimeIndex
        as the input data (or by the inex of the user-supplied series).

        Parameters
        ----------
        algo : str or pandas.Series, optional
            Sleep scoring method to use or pandas Series to use as sleep vector
            - If a string, must be one of:
                {"Roenneberg", "Crespo", "Oakley", "Scripps", "Sadeh", "Cole_Kripke"}.

            - If a pandas Series, it is assumed to already represent a sleep/rest
              vector (binary) and is returned unchanged. This is useful when the
              user has computed sleep elsewhere or wants full control
        kwargs
            Additional keyword arguments passed through the selected algorithm.
            For example, you can override algorithm default parameters such as
            thresholds, windows, or rescoring options.

        Returns
        -------
        pandas.Series
            Binary sleep/rest classification time series.

            Convention: values are expected to encode sleep/rest as 1 and wake as 0.
            If an external algorithm uses the opposite convention, convert it before
            passing it here.

        Raises
        ------
        ValueError
            If self.data is not indexed by a DatetimeIndex (needed to infer the sampling
            interval), or if algo is an unknown algorithm.
        """
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex.")

        # Estimate sampling interval (minutes)
        dt = self.data.index.to_series().diff().dropna().median()
        freq_minutes = dt.total_seconds() / 60

        match algo:
            case 'Roenneberg':
                return Roenneberg(data=self.data, plot=False, **kwargs)
            case 'Crespo':
                return Crespo(
                    data=self.data,
                    frequency=freq_minutes,
                    plot=False,
                    **kwargs
                )
            case 'Oakley':
                return Oakley(data=self.data, **kwargs)
            case 'Scripps':
                scripps = Scripps(data=self.data, **kwargs)
                return scripps
            case 'Sadeh':
                return Sadeh(data=self.data, **kwargs)
            case 'Cole_Kripke':
                return Cole_Kripke(data=self.data, **kwargs)
            case _:
                if isinstance(algo, pd.Series):
                    return algo
                raise ValueError(f'Unknown algo {algo}')

    def derivative(self, t, state, input):
        """
        Computes the derivatives of the state variables at a given time, sleep and light input.

        Parameters
        ----------
        t : float
            Current simulation time (in hours).
        state : numpy.ndarray
            Current values of the state variables at time t.
        input: tuple
            Contains light intensity input (in lux) at time t and the corresponding wake state.

        Returns
        -------
        numpy.ndarray
            Derivatives at time t.
        """
        x = state[0]
        xc = state[1]
        n = state[2]
        light = input
        wake = self.sleep

        # note this correction on the alpha term (important to test Forger's model
        # better approximates Hannay's under the ModelComparer framework)
        alpha = self.a0 * np.power(light / self.i0, self.p) * (light / (light + 100))
        bhat = self.g * (1 - n) * alpha * (1 - (0.4 * x)) * (1 - 0.4 * xc)

        # Sigma is set to one if rest/sleep
        sigma = 1.0 if wake else 0.0

        # Calculate psi_cx
        c = t % 24
        cbtmin = self.phi_xcx + self.phi_ref
        cbtminlocal = (cbtmin * 24.0) / (2 * np.pi)
        psi_cx = (c - cbtminlocal) % 24

        # Calculate Ns
        if 16.5 < psi_cx < 21.0:
            nsh = self.rho * (1.0 / 3.0)
        else:
            nsh = self.rho * ((1.0 / 3.0) - sigma)
        ns = nsh * (1 - np.tanh(10.0 * x))

        mu_term = self.mu * (
            (1.0 / 3.0) * x
            + (4.0 / 3.0) * np.power(x, 3.0)
            - (256.0 / 105.0) * np.power(x, 7.0)
        )

        taux_term = np.power((24.0 / (0.99729 * self.taux)), 2.0) + self.k * bhat

        # Differential equation system
        dydt = np.zeros_like(state)
        dydt[0] = np.pi / 12.0 * (xc + mu_term + bhat + ns)
        dydt[1] = np.pi / 12.0 * (self.q * bhat * xc - x * taux_term)
        dydt[2] = 60.0 * (alpha * (1.0 - n) - self.beta * n)

        return dydt

    def amplitude(self):
        """
        Amplitude of the oscillator computed from the integrated state trajectory.

        Returns
        -------
        numpy.ndarray
            Amplitude at each time point: sqrt(x^2 + xc^2).
        """
        # Integrate model and extract amplitude
        x = self.model_states[:, 0]
        xc = self.model_states[:, 1]
        return np.sqrt(x**2 + xc**2)

    def phase(self):
        """
        Returns the phase angle of the oscillator computed from the integrated state trajectory.

        Returns
        -------
        numpy.ndarray
            Phase angle (radians) at each time point: arctangent(xc/x).
        """
        # Integrate model and extract phase
        x = self.model_states[:, 0]
        # Multiplying xc makes the phase move clockwise
        xc = -1.0 * self.model_states[:, 1]
        # Observe that np.angle(x + complex(0,1) * xc) == np.atan2(xc,x)
        # The y (in this case, xc) appears first in the atan2 formula
        return np.angle(x + complex(0, 1) * xc)
        # return np.atan2(xc,x)

    def cbt(self):
        """
        Time points corresponding to the predicted core bod temperature minima (CBTmin), derived from the
        state variable x.

        Returns
        -------
        numpy.ndarray
            Array of time points (in hours) where minima of x occur, corresponding to the CBTmin.
        """
        # Calculate time step (dt) between consecutive time points
        dt = np.diff(self.time)[0]
        # Invert cos(x) to turn the minima into maxima (peaks)
        inverted_x = -1.0 * self.model_states[:, 0]
        # Identify the indices where the minima occur
        cbt_min_indices, _ = find_peaks(inverted_x, distance=np.ceil(13.0 / dt))
        # Use the previous indices to find the cbtmin times
        cbtmin_times = self.time[cbt_min_indices] + self.phi_ref
        # if you want to know in clock time, just do cbtmin_times % 24
        return cbtmin_times


class Breslow13(Model):
    """
    Implements the mathematical model of human circadian rhythms developed by Forger, Jewett and Kronauer [1].
    The formalism includes a representation of the biochemical conversion of the light signal into a drive on
    the circadian pacemaker, which is modeled as a van der Pol oscillator. This cubic model is characterized
    by three state variables: x, xc and n. While n can be interpreted as the proportion of activated photoreceptors,
    at a given time, x and xc cannot directly be mapped to specific physiological mechanisms. Instead, x and xc are
    used to predict biologically meaningful quantities, such as the core body temperature minimum (CBTmin).

    Our implementation closely follows the approach of the `circadian`package by Arcascope [2]. However, we use the
    more powerful LSODA integrator (via SciPy's `odeint`) for numerical integration, enabling the integration of the
    system using more complex light trajectories.

    Attributes
    ----------
    taux : float
        Intrinsic period of the oscillator (hours).
    mu : float
        stiffness of the van der Pol oscillator.
    g : float
        Light sensitivity scaling parameter.
    alpha_0 : float
        Baseline forward rate constant governing the photon-driven activation
        (or depletion) of photoreceptors. This parameter scales the rate at which
        incident light converts available photoreceptors into an active or
        “used” state.
    beta : float
        Backward rate constant governing photoreceptor recovery or regeneration.
        This parameter controls the rate at which used photoreceptors return to
        the available state, representing dark adaptation or biochemical
        recovery processes.
    p : float
        Power-law exponent for light-dependent photoreceptor activation. This
        parameter determines how sensitively the forward rate constant α scales
        with light intensity.
    i0 : float
        Reference light intensity used to normalize the input light intensity vector I.
    k : float
        Coupling coefficient scaling the influence of the photic drive B on the
        circadian pacemaker. It modulates the sensitivity of the oscillator to light.
    cbt_to_dlmo : float
        Time offset (in hours) from CBTmin to DLMO.
    initial_conditions : numpy.ndarray
        State vector at the start of simulation (default: [-0.0843259, -1.09607546, 0.45584306]).
    model_states : numpy.ndarray
        Integrated state trajectories of the model.
    time : numpy.ndarray
        Array of time points for simulation.
    inputs : numpy.ndarray
        Array of input values (e.g., light intensity and wake) over time.

    Methods
    -------
    derivative(t, state, light)
        Computes the derivatives of the state variables at a given time and light input.
    amplitude()
        Calculates the amplitude of the oscillator from integrated states.
    phase()
        Calculates the phase angle of the oscillator from integrated states.
    cbt()
        Identifies the timing of core body temperature minima from integrated states.

    References
    ----------
    [1] Forger DB, Jewett ME, Kronauer RE. A Simpler Model of the Human Circadian Pacemaker.
    Journal of Biological Rhythms. 1999;14(6):533-538. doi:10.1177/074873099129000867

    [2] Tavella, F., Hannay, K., & Walch, O. (2023). Arcascope/circadian: Refactoring of readers
    and metrics modules, Zenodo, v1.0.2. https://doi.org/10.5281/zenodo.8206871
    """

    def __init__(
        self,
        data=None,
        inputs=None,
        time=None,
        h1_threshold=0.001,
        k=0.55,
        i0=9500.0,
        i1=100.0,
        alpha_0=0.1,
        beta=0.007,
        p=0.5,
        r=15.36,
        g=37.0,
        b=0.4,
        gamma=0.13,
        kappa=12.0 / np.pi,
        tauc=24.1,
        f=0.99729,
        m=7.0,
        eta=0.04,
        xi=0.54,
        beta_ip=0.000783 * 3600.0,
        beta_cp=0.000335 * 3600.0,
        beta_ap=0.000162 * 3600.0,
        a=0.0010422 * 3600.0,
        phi_on=6.113 - 2 * np.pi,
        phi_off=4.352 - 2 * np.pi,
        delta=600.0,
        mmax=0.019513,
        hstat=861.0,
        sigma=50.0,
        phi_ref=0.97,
        cbt_to_dlmo=7.0,
        initial_condition=None,
    ):
        if inputs is None or time is None:
            super().__init__(
                data=data,
                initial_conditions=np.array([-0.219, -1.22, 0.519, 0.0, 0.0, 0.0]),
            )
        else:
            super().__init__(
                inputs=inputs,
                time=time,
                initial_conditions=np.array([-0.219, -1.22, 0.519, 0.0, 0.0, 0.0]),
            )
        # Check for a scenario in which a initial condition is provided
        if initial_condition is not None:
            self.initial_conditions = initial_condition
        self.model_name = self.__class__.__name__
        self.h1_threshold = h1_threshold
        self.k = k
        self.i0 = i0
        self.i1 = i1
        self.alpha_0 = alpha_0
        self.beta = beta
        self.p = p
        self.r = r
        self.g = g
        self.b = b
        self.gamma = gamma
        self.kappa = kappa
        self.tauc = tauc
        self.f = f
        self.m = m
        self.eta = eta
        self.xi = xi
        self.beta_ip = beta_ip
        self.beta_cp = beta_cp
        self.beta_ap = beta_ap
        self.a = a
        self.phi_on = phi_on
        self.phi_off = phi_off
        self.delta = delta
        self.mmax = mmax
        self.hstat = hstat
        self.sigma = sigma
        self.phi_ref = phi_ref
        self.cbt_to_dlmo = cbt_to_dlmo
        self.initialize_model_states()

    def avoid_negative_h1(self, h1, b):
        if h1 < self.h1_threshold and (1.0 - self.m * b) < 0:
            return 0.0
        else:
            return 1.0

    def melatonin_drive(self, h2):
        exp_factor = 1.0 + np.exp((self.hstat - h2) / self.sigma)
        return self.mmax / exp_factor

    def derivative(self, t, state, input):
        """
        Computes the derivatives of the state variables at a given time and light input.

        Parameters
        ----------
        t : float
            Current simulation time (in hours).
        state : numpy.ndarray
            Current values of the state variables at time t.
        input: float
            Contains light intensity input (in lux) at time t and the corresponding wake state.

        Returns
        -------
        numpy.ndarray
            Derivatives at time t.
        """
        x = state[0]
        xc = state[1]
        n = state[2]
        h1 = state[3]  # pineal melatonin
        h2 = state[4]  # plasma melatonin
        h3 = state[5]  # exogenous melatonin
        light = input

        alpha = (
            self.alpha_0 * ((light / self.i0) ** self.p) * (light / (light + self.i1))
        )
        b = self.g * alpha * (1.0 - n) * (1.0 - self.b * x) * (1.0 - self.b * xc)
        gamma_term = self.gamma * (x / 3.0 + 4.0 / 3.0 * x**3 - 256.0 / 105.0 * x**7)
        tauc_term = pow(24.0 / (self.f * self.tauc), 2) + self.k * b
        m = self.melatonin_drive(h2)
        s_h1_b = self.avoid_negative_h1(h1, b)

        phase = np.arctan2(
            x, xc
        )  # note this is the opposite to all other models (here xc is on the x-axis)
        if self.phi_on > phase > self.phi_off:
            activation = 1.0 - np.exp(
                -self.delta * np.mod(self.phi_on - phase, 2 * np.pi)
            )
            normalization = 1.0 - np.exp(
                -self.delta * np.mod(self.phi_on - self.phi_off, 2 * np.pi)
            )
            a = self.a * activation / normalization
        else:
            a = self.a * np.exp(-self.r * np.mod(self.phi_on - self.phi_off, 2 * np.pi))

        # Differential equation system
        dydt = np.zeros_like(state)
        dydt[0] = (1.0 / self.kappa) * (xc + gamma_term + b - self.eta * m)
        dydt[1] = (1.0 / self.kappa) * (b * xc / 3.0 - x * tauc_term - self.xi * m)
        dydt[2] = 60.0 * (alpha * (1 - n) - self.beta * n)  # typo on paper missing 60.0
        dydt[3] = -self.beta_ip * h1 + a * (1.0 - self.m * b) * s_h1_b
        dydt[4] = self.beta_ip * h1 - self.beta_cp * h2 + self.beta_ap * h3
        dydt[5] = -self.beta_ap * h3

        return dydt

    def amplitude(self):
        """
        Amplitude of the oscillator computed from the integrated state trajectory.

        Returns
        -------
        numpy.ndarray
            Amplitude at each time point: sqrt(x^2 + xc^2).
        """
        # Integrate model and extract amplitude
        x = self.model_states[:, 0]
        xc = self.model_states[:, 1]
        return np.sqrt(x**2 + xc**2)

    def phase(self):
        """
        Returns the phase angle of the oscillator computed from the integrated state trajectory.

        Returns
        -------
        numpy.ndarray
            Phase angle (radians) at each time point: arctangent(xc/x).
        """
        # Integrate model and extract phase
        x = self.model_states[:, 0]
        # Multiplying xc makes the phase move clockwise
        xc = -1.0 * self.model_states[:, 1]
        # Observe that np.angle(x + complex(0,1) * xc) == np.atan2(xc,x)
        # The y (in this case, xc) appears first in the atan2 formula
        return np.angle(x + complex(0, 1) * xc)
        # return np.atan2(xc,x)

    def cbt(self):
        """
        Time points corresponding to the predicted core bod temperature minima (CBTmin), derived from the
        state variable x.

        Returns
        -------
        numpy.ndarray
            Array of time points (in hours) where minima of x occur, corresponding to the CBTmin.
        """
        # Calculate time step (dt) between consecutive time points
        dt = np.diff(self.time)[0]
        # Invert cos(x) to turn the minima into maxima (peaks)
        inverted_x = -1.0 * self.model_states[:, 0]
        # Identify the indices where the minima occur
        cbt_min_indices, _ = find_peaks(inverted_x, distance=np.ceil(13.0 / dt))
        # Use the previous indices to find the cbtmin times
        cbtmin_times = self.time[cbt_min_indices] + self.phi_ref
        # if you want to know in clock time, just do cbtmin_times % 24
        return cbtmin_times


class Skeldon23:
    """
    Under construction
    """

    def __init__(
        self,
        data=None,
        inputs=None,
        time=None,
        mu=17.78,
        chi=45.0,
        h0=13.0,
        delta=1.0,
        ca=1.72,
        tauc=24.2,
        f=0.99669,
        g=19.9,
        p=0.6,
        k=0.55,
        b=0.4,
        gamma=0.23,
        alpha_0=0.16,
        beta=0.013,
        i0=9500.0,
        kappa=24.0 / (2.0 * np.pi),
        c20=0.7896,
        alpha21=-0.3912,
        alpha22=0.7583,
        beta21=-0.4442,
        beta22=0.0250,
        beta23=-0.9647,
        s0=0.0,
        forced_wakeup_light_threshold=None,
        forced_wakeup_weekday_only=False,
        cbt_to_dlmo=7.0,
        initial_condition=np.array([0.23995682, -1.1547196, 0.50529415, 12.83846474]),
    ):
        self.sleep_state = None
        self.received_light = None
        self.model_name = self.__class__.__name__
        self.initial_conditions = initial_condition
        self.model_states = None
        self.data = data

        # -------------------------------------
        # Input handling logic (time and light)
        # -------------------------------------

        # Extract time from data index
        if time is None or inputs is None:
            if data is not None:
                self.time = np.asarray(
                    (data.index - data.index.min()).total_seconds() / 3600
                )
                self.inputs = np.asarray(data.values)
            else:
                raise ValueError(
                    "Must provide either light time series (data) or input and time."
                )
        else:
            self.time = time
            self.inputs = inputs

        # ----------------
        # Model parameters
        # ----------------

        # Sleep/wake regulation parameters
        self.mu = mu
        self.chi = chi
        self.h0 = h0
        self.delta = delta
        self.ca = ca

        # Light/circadian parameters
        self.tauc = tauc
        self.f = f
        self.g = g
        self.p = p
        self.k = k
        self.b = b
        self.gamma = gamma
        self.alpha_0 = alpha_0
        self.beta = beta
        self.i0 = i0
        self.kappa = kappa

        # Parameters that perform circadian modulatation of wakefulness
        self.c20 = c20
        self.alpha21 = alpha21
        self.alpha22 = alpha22
        self.beta21 = beta21
        self.beta22 = beta22
        self.beta23 = beta23

        # Sleep state (0 for wake, 1 for sleep) - first sleep state given by user
        self.current_sleep_state = s0

        # Forced wake up
        self.forced_wakeup_light_threshold = forced_wakeup_light_threshold
        self.forced_wakeup_weekday_only = forced_wakeup_weekday_only

        self.cbt_to_dlmo = cbt_to_dlmo

    def dlmos(self):
        """
        Calculates the predicted Dim Light Melatonin Onset (DLMO) time points using a fixed,
        pre-specified offset (default: seven hours).

        Returns
        -------
        numpy.ndarray:
            Array of time points (in hours) at which DLMO is expected to occur.
        """
        return self.cbt() - self.cbt_to_dlmo

    def get_initial_conditions(
        self, loop_number, data=None, light_vector=None, time_vector=None
    ):
        """
        Attempts to equilibrate the model's initial conditions by repeated simulation.

        Works with the piecewise odeint integrator that returns:
        (time_hours, state_trajectory, sleep_state_trajectory)

        Returns
        -------
        np.ndarray
            Final continuous state [x, xc, n, h] after entrainment attempt.
        """
        # Extract time from light index
        if time_vector is None or light_vector is None:
            if data is not None:
                time_vector = np.asarray(
                    (data.index - data.index.min()).total_seconds() / 3600
                )
                light_vector = np.asarray(data.values)
            else:
                raise ValueError("Must provide either light series or input and time.")
        else:
            time_vector = np.asarray(time_vector, dtype=float)
            light_vector = np.asarray(light_vector, dtype=float)

        # Save existing inputs/time to prevent being permanently overwritten
        old_time = getattr(self, "time", None)
        old_inputs = getattr(self, "inputs", None)

        # Initialize with default initial conditions
        initial_condition = np.asarray(self.initial_conditions, dtype=float)

        # List to collect dlmos
        dlmos = []

        try:
            for i in range(loop_number):
                # Calculate the solution for the models
                self.time = time_vector
                self.inputs = light_vector

                # Run one full cycle simulation starting from current initial_conditions
                time_hours, state_trajectory, sleep_state_trajectory = (
                    self.integrate_piecewise_odeint(initial_state=initial_condition)
                )

                self.model_states = state_trajectory

                # Compute dlmos from the solution
                dlmos.append(self.dlmos())

                # Update initial_condition for the next iteration
                initial_condition = state_trajectory[-1]

                # Exit early if entrainment is detected
                if i > 1 and np.isclose(dlmos[-1][-1], dlmos[-2][-1], atol=1e-3):
                    # Print number of loops required for entrainment
                    # print(f"The model entrained after {i} loops.")
                    # Update model initial conditions to entrained state
                    self.initial_conditions = state_trajectory[-1]

                    # Return entrained model solution
                    return state_trajectory[-1]

            # Return unentrained model solution
            self.initial_conditions = state_trajectory[-1]
            return state_trajectory[-1]
        finally:
            # Restore whatever was previously on the model
            self.time = old_time
            self.inputs = old_inputs

    # ---------------------------------------------------
    # Switching between awake and asleep (discrete steps)
    # ---------------------------------------------------

    def _circadian_modulation_C(self, x, xc):
        """
        Circadian modulation term C(x,xc) used in the sleep switching threshold
        """
        linear_term = self.c20 + self.alpha21 * xc + self.alpha22 * x
        quadratic_term = (
            self.beta21 * xc * xc + self.beta22 * xc * x + self.beta23 * x * x
        )
        return linear_term + quadratic_term

    def _is_weekday(self, t_hours: float) -> bool:
        """
        Author's weekday rule: treat t=0 as start of a 7-day cycle;
        weekday if (t/24) % 7 < 5 (5 is Saturday)
        """
        return ((t_hours / 24.0) % 7) < 5

    def update_sleep_state(
        self, t: float, state: np.ndarray, light_input: float
    ) -> float:
        """
        Authors' post-step sleep-state update rule.

        This method should be called exactly once per external time step, after
        integrating the continuous ODE across that step.

        """
        x = state[0]
        xc = state[1]
        h = state[3]

        s = float(self.current_sleep_state)

        if s not in (0.0, 1.0):
            raise ValueError("current_sleep_state must be 0 or 1")

        c = self._circadian_modulation_C(x, xc)

        new_s = s
        if s == 0.0:
            # wake -> sleep
            h_plus = self.h0 + 0.5 * self.delta + self.ca * c
            if h >= h_plus:
                new_s = 1.0
        else:
            # sleep -> wake
            h_minus = self.h0 - 0.5 * self.delta + self.ca * c
            if h <= h_minus:
                new_s = 0.0

        # Forced wake-up by light (mirror authors' behavior)
        if self.forced_wakeup_light_threshold is not None:
            weekday_ok = True
            if self.forced_wakeup_weekday_only:
                weekday_ok = self._is_weekday(t)

            if weekday_ok and (light_input > self.forced_wakeup_light_threshold):
                new_s = 0.0

        self.current_sleep_state = new_s
        return new_s

    # -----------------------------
    # Integration step using odeint
    # -----------------------------

    def integrate_piecewise_odeint(self, initial_state=None):
        """
        Integrate the Skeldon23 sleep/circadian model over time.

        This model contains:
          - Continuous variables (circadian + sleep pressure)
          - A discrete sleep state (awake=0, asleep=1)

        Because sleep state is discrete, we:
          1) Keep sleep state fixed within each time interval
          2) Integrate the continuous equations across that interval
          3) Update sleep state at the end of the interval

        Returns
        -------
        time_hours : np.ndarray
            Time values in hours.
        state_trajectory : np.ndarray
            Continuous state history [x, xc, n, h].
        sleep_state_history : np.ndarray
            Sleep/wake state over time (0=wake, 1=sleep).
        """
        # Convert time and light input to numpy arrays
        time_hours = np.asarray(self.time, dtype=float)
        light_input = np.asarray(self.inputs)

        # Initial continuous state
        if initial_state is None:
            current_state = np.asarray(self.initial_conditions, dtype=float)
        else:
            current_state = np.asarray(initial_state, dtype=float)

        # Number of timepoints in the total state trajectory
        n_timepoints = len(time_hours)

        # Store full trajectory of state variables
        state_trajectory = np.zeros((n_timepoints, 4), dtype=float)

        # Store full trajectory of sleep states (0=wake,1=sleep)
        sleep_state_trajectory = np.zeros(n_timepoints, dtype=float)

        # Set initial values
        state_trajectory[0] = current_state
        sleep_state_trajectory[0] = self.current_sleep_state

        # Loop over each time interval
        for i in range(n_timepoints - 1):
            # Define start and end times
            start_time = float(time_hours[i])
            end_time = float(time_hours[i + 1])

            # Assume that light is constant within this interval
            interval_light = float(light_input[i])

            # Freeze sleep state during this interval
            sleep_state_fixed = float(self.current_sleep_state)

            def derivative(state, tt):
                """
                Model master equations.
                """
                # Model state equations (no direct physiological interpretation)
                x = state[0]
                xc = state[1]

                # Photic drive
                n = state[2]

                # Sleep pressure
                h = state[3]

                # Assume that light only affects the system when awake
                light = (1.0 - sleep_state_fixed) * interval_light

                # Convert light into biological drive
                if light <= 0:
                    alpha = 0.0
                else:
                    alpha = self.alpha_0 * (light / self.i0) ** self.p

                # Photic input to the circadian system
                b = (
                    self.g
                    * (1.0 - n)
                    * alpha
                    * (1.0 - self.b * x)
                    * (1.0 - self.b * xc)
                )

                # Oscillator dynamics
                gamma_term = self.gamma * (xc - 4.0 / 3.0 * xc**3)
                tauc_term = (24.0 / (self.f * self.tauc)) ** 2 + self.k * b

                # Define derivatives
                dydt = np.zeros_like(state)
                dydt[0] = (1.0 / self.kappa) * (xc + b)
                dydt[1] = (1.0 / self.kappa) * (gamma_term - x * tauc_term)
                dydt[2] = 60.0 * (alpha * (1.0 - n) - self.beta * n)

                # Sleep pressure increases when awake, decreases when asleep
                dydt[3] = (1.0 / self.chi) * (-h + (1.0 - sleep_state_fixed) * self.mu)

                return dydt

            # Integrate system across interval
            t_interval = [start_time, end_time]
            solution = odeint(derivative, state_trajectory[i], t_interval)

            new_state = solution[-1]
            state_trajectory[i + 1] = new_state

            # Update sleep state AFTER integration
            new_sleep_state = self.update_sleep_state(
                end_time, new_state, interval_light
            )

            sleep_state_trajectory[i + 1] = new_sleep_state

        return time_hours, state_trajectory, sleep_state_trajectory

    def amplitude(self):
        """
        Amplitude of the oscillator computed from the integrated state trajectory.

        Returns
        -------
        numpy.ndarray
            Amplitude at each time point: sqrt(x^2 + xc^2).
        """
        # Integrate model and extract amplitude
        x = self.model_states[:, 0]
        xc = self.model_states[:, 1]
        return np.sqrt(x**2 + xc**2)

    def phase(self):
        """
        Returns the phase angle of the oscillator computed from the integrated state trajectory.

        Returns
        -------
        numpy.ndarray
            Phase angle (radians) at each time point: arctangent(xc/x).
        """
        # Integrate model and extract phase
        x = self.model_states[:, 0]
        # Multiplying xc makes the phase move clockwise
        xc = -1.0 * self.model_states[:, 1]
        # Observe that np.angle(x + complex(0,1) * xc) == np.atan2(xc,x)
        # The y (in this case, xc) appears first in the atan2 formula
        return np.angle(x + complex(0, 1) * xc)
        # return np.atan2(xc,x)

    def cbt(self):
        """
        Time points corresponding to the predicted core bod temperature minima (CBTmin), derived from the
        state variable x.

        Returns
        -------
        numpy.ndarray
            Array of time points (in hours) where minima of x occur, corresponding to the CBTmin.
        """
        # Calculate time step (dt) between consecutive time points
        dt = np.diff(self.time)[0]
        # Invert cos(x) to turn the minima into maxima (peaks)
        inverted_x = -1.0 * self.model_states[:, 0]
        # Identify the indices where the minima occur
        cbt_min_indices, _ = find_peaks(inverted_x, distance=np.ceil(13.0 / dt))
        # Use the previous indices to find the cbtmin times
        cbtmin_times = self.time[cbt_min_indices]
        # if you want to know in clock time, just do cbtmin_times % 24
        return cbtmin_times


def main():
    # Parameters for the light schedule
    total_days = 10  # Number of days
    light_on_hours = 16  # Hours lights are on
    light_off_hours = 8  # Hours lights are off
    bins_per_hour = 6  # 10-minute bins per hour
    # bins_per_hour = 10
    # bins_per_hour = 1

    # Total bins for light on and off periods
    light_on_bins = light_on_hours * bins_per_hour
    light_off_bins = light_off_hours * bins_per_hour

    # Generate random light levels for the light-on period (e.g., between 100 and 800 lux)
    np.random.seed(42)
    light_on_variation = np.random.randint(low=900, high=901, size=light_on_bins)

    # Create the daily schedule: light-off period (0 lux) followed by light-on period
    daily_schedule = np.concatenate([np.zeros(light_off_bins), light_on_variation])

    # Repeat for the total number of days
    light = np.tile(daily_schedule, total_days)

    dt = 10 / 60  # 10 minutes in hours
    # dt = 1
    # dt=1/10 # for 10 bins/h
    time = np.arange(0, len(light) * dt, dt)

    # SKELDON
    skeldon = Skeldon23(inputs=light, time=time)
    initial = skeldon.get_initial_conditions(
        loop_number=10, light_vector=light, time_vector=time
    )

    t, states, sleep = skeldon.integrate_piecewise_odeint(initial_state=initial)

    x = states[:, 0]
    xc = states[:, 1]
    n = states[:, 2]
    h = states[:, 3]

    plt.figure(figsize=(15, 8))
    plt.plot(t, x)
    plt.ylabel("var")
    plt.show()

    return None

    # SECTION FOR COMPARING CIRCADIAN MODELS (FORGER AND HANNAY)
    comparison = ModelComparer(inputs=light, time=time, equilibrate=True)
    comparison.linearize_phase(change_params=True)
    comparison.find_optimal_params(change_params=True)
    comparison.predict_forger(change_params=True)
    # comparison.error(change_params=True)
    print(comparison.rmse())
    # max_error_x, min_error_x, error_band_x, magnitude_x = comparison.error_stats()[0]
    # max_error_xc, min_error_xc, error_band_xc, magnitude_xc = comparison.error_stats()[1]

    # SECTION WITH GENERAL COMMANDS
    # hannay = HannaySP(inputs=light, time=time)
    # forger = HannaySP(inputs=light, time=time)
    # esri = ESRI(inputs=light, time=time)
    # esri.calculate()

    # start = time.time()
    # r = forger.model_states[:,1]
    # states = list(r)
    # r = jewett.integrate()[:,0]
    # r = hannay.amplitude()
    # r = hannay.phase()
    # x = forger.integrate()[:,1]
    # print(hannay.dlmos() % 24)
    # phi = hannay.integrate()[:, 1]
    # predicted_x = hannay.predict_forger()[0]
    # hannay.initial_conditions = np.array([1,2,3])
    # print(forger.equilibrate(100, light_vector, time_vector))
    # print(hannay.equilibrate(100, light_vector, time_vector))
    # end = time.time()

    # REGION FOR PLOTS
    # plt.figure(figsize=(18, 8))
    # plt.plot(phi)
    # plt.xlabel("Time")
    # plt.ylabel("Phi (state variable)")
    # plt.title("State Evolution Over Time")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # plt.figure()
    # plt.plot(comparison.predicted_x, label="predicted_x", color="black", alpha=0.5)
    # plt.plot(comparison.predicted_x, label="predicted_x", color="blue", alpha=0.9)
    # peaks,_= find_peaks(comparison.error_x)
    # for peak in peaks:
    #   print(f'{comparison.error_x[peak]:.3f}')
    # plt.plot(comparison.error_x, label="Error", color="pink", alpha=0.9)
    # plt.axhline(y=max_error_x, color="green", linestyle="--", label="Max Error")
    # plt.axhline(y=min_error_x, color="purple", linestyle="--", label="Min error")
    # plt.text(
    #   x=len(comparison.x) // 2,
    #  y=(max_error_x + min_error_x) / 2,
    # s=f"Error band width {error_band_x:.2f}",
    # color="orange",
    # ha="center",
    # va="center",
    # fontsize=12,
    # backgroundcolor="white",
    # )
    # plt.xlabel("Time")
    # plt.ylabel("State variable")
    # plt.title("State Evolution Over Time")
    # plt.legend()
    # plt.grid(True)
    # print(f"Magnitude of the error in predicting x = {magnitude_x}")

    # plt.figure()
    # plt.plot(comparison.predicted_xc, label="predicted_xc", color="black", alpha=0.5)
    # plt.plot(comparison.xc, label="observed_xc", color="blue", alpha=0.9)
    # plt.plot(comparison.error_xc, label="Error", color="pink", alpha=0.9)
    # plt.axhline(y=max_error_xc, color="green", linestyle="--", label="Max Error")
    # plt.axhline(y=min_error_xc, color="purple", linestyle="--", label="Min error")
    # plt.text(
    #   x=len(comparison.xc) // 2,
    #  y=(max_error_xc + min_error_xc) / 2,
    # s=f"Error band width {error_band_xc:.2f}",
    # color="orange",
    # ha="center",
    # va="center",
    # fontsize=12,
    # backgroundcolor="white",
    # )
    # plt.xlabel("Time")
    # plt.ylabel("State variable")
    # plt.title("State Evolution Over Time")
    # plt.legend()
    # plt.grid(True)
    # print(f"Magnitude of the error in predicting x = {magnitude_xc}")

    # plt.figure(figsize=(18,8))
    # plt.plot(comparison.error_x, label='Error', color='pink', alpha=0.9)
    # plt.xlabel("Time")
    # plt.ylabel("Error in x prediction")
    # plt.title("x Error Evolution Over Time")
    # plt.legend()
    # plt.grid(True)

    # plt.figure(figsize=(18,8))
    # plt.plot(comparison.error_xc, label='Error', color='pink', alpha=0.9)
    # plt.xlabel("Time")
    # plt.ylabel("Error in xc prediction")
    # plt.title("xc Error Evolution Over Time")
    # plt.legend()
    # plt.grid(True)
    # plt.figure()
    # plt.plot(r, label=f'State r', color='blue', alpha=0.7)
    # plt.xlabel("Time")
    # plt.ylabel(f"Model state r")
    # plt.title(f"Model State r Evolution Over Time")
    # plt.legend()
    # plt.grid(True)

    # plt.show()

    # print(f"Execution time is {end-start} seconds.")


if __name__ == "__main__":
    main()
