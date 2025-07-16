import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


class Model:
    """
    Numerically integrates a system of ordinary differential equations (ODEs), whose evolution may
    depend on external inputs (e.g., light) and initial conditions. Time points and input data are
    either provided directly or extracted from a pandas.Series with a DatetimeIndex.

    Attributes
    ----------
        initial_conditions (numpy.ndarray):
            Array of initial conditions for the model states.
        model_states (numpy.ndarray or None):
            Array containing the model state trajectories after integration.
        data (pandas.Series, optional):
            Input data series with a DatetimeIndex, where the index specifies the time points and
            the values represent the input variable (e.g., light intensity). Time and value arrays
            are extracted from this series.
        time (numpy.ndarray, optional):
            Array of time points (in hours); must be monotonically increasing.
        inputs (numpy.ndarray, optional):
            Array of input values (e.g., light intensity) corresponding to each time point.
    """
    def __init__(self, initial_conditions, data=None, time=None, inputs=None):
        self.initial_conditions = initial_conditions
        self.model_states = None
        self.data = data

        # Extract time from data index
        if time is None or inputs is None:
            if data is not None:
                self.time = np.asarray((data.index - data.index.min()).total_seconds() / 3600)
                self.inputs = np.asarray(data.values)
            else:
                raise ValueError("Must provide either light time series (data) or input and time.")
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

        This method uses SciPy `odeint` function to simulate the model dynamics over time,
        given a set of initial conditions and external inputs (e.g., light intensity).
        By default, it uses the class attributes `self.inputs`, `self.time`, and `self.initial_condition`,
        but alternative arrays can be provided.

        The input and time vectors must be of the same length, and the system's equations must be defined
        by the `derivative`method (which should be implemented within subclass of the Model class).

        Parameters
        ----------
        light_vector (numpy.ndarray, optional):
            Array of input values (typically, light intensity)
        time_vector (numpy.ndarray, optional):
            Array of time points (typically, time in hours)
        initial_condition (numpy.ndarray, optional):
            Array of initial conditions

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


    def get_initial_conditions(self, loop_number, data=None, light_vector=None, time_vector=None):
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
            Input data series with a DatetimeIndex. The index represents time points
            and the values represent input data (e.g., light intensity). The time
            and input arrays are extracted from this series.
        light_vector (numpy.ndarray, optional):
            Array of input values (typically, light intensity)
        time_vector (numpy.ndarray, optional):
            Array of time points (typically, time in hours)

        Returns
        ----------
        numpy.ndarray:
            The final state of the model after attempting to reach equilibrium (either an equilibrated solution or last
            simulated state).
        """
        # Extract time from light index
        if time_vector is None or light_vector is None:
            if data is not None:
                time_vector = np.asarray((data.index - data.index.min()).total_seconds() / 3600)
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
                print(f"The model entrained after {i} loops.")
                # Update model initial conditions to entrained state
                self.initial_conditions = solution[-1]

                # Return entrained model solution
                return solution[-1]
        # Non-entrainment message(free-running rhythm)
        print(
            "The model did not entrain due to insufficient loops or unentrainable light schedule."
        )
        # Return unentrained model solution
        self.initial_conditions = solution[-1]
        return solution[-1]


    def dlmos(self):
        return self.cbt() - self.cbt_to_dlmo


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
        Nonlinearity parameter of the van der Pol oscillator.
    g : float
        Light sensitivity scaling parameter.
    alpha_0 : float
        Baseline light sensitivity parameter.
    beta : float
        Rate constant for adaptation variable n.
    p : float
        Power-law exponent for light input.
    i0 : float
        Saturation intensity for light sensitivity (lux).
    k : float
        Scaling factor for light-dependent changes in oscillator period.
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
                initial_conditions=np.array([-0.0843259, -1.09607546, 0.45584306])
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
        # Integrate model and extract amplitude
        x = self.model_states[:, 0]
        xc = self.model_states[:, 1]
        return np.sqrt(x**2 + xc**2)

    def phase(self):
        # Integrate model and extract phase
        x = self.model_states[:, 0]
        # Multiplying xc makes the phase move clockwise
        xc = -1.0 * self.model_states[:, 1]
        # Observe that np.angle(x + complex(0,1) * xc) == np.atan2(xc,x)
        # The y (in this case, xc) appears first in the atan2 formula
        return np.angle(x + complex(0, 1) * xc)
        # return np.atan2(xc,x)

    def cbt(self):
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
        Nonlinearity parameter of the van der Pol oscillator.
    g : float
        Light sensitivity scaling parameter.
    alpha_0 : float
        Baseline light sensitivity parameter.
    beta : float
        Rate constant for adaptation variable n.
    p : float
        Power-law exponent for light input.
    i0 : float
        Saturation intensity for light sensitivity (lux).
    k : float
        Scaling factor for light-dependent changes in oscillator period.
    q : float
        Coefficient for light-dependent feedback in xc dynamics.
    phi_ref : float
        Reference phase parameter for phase computations.
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
                initial_conditions=np.array([-0.10097101, -1.21985662, 0.50529415])
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
        # Integrate model and extract amplitude
        x = self.model_states[:, 0]
        xc = self.model_states[:, 1]
        return np.sqrt(x**2 + xc**2)

    def phase(self):
        # Integrate model and extract phase
        x = self.model_states[:, 0]
        # Multiplying xc makes the phase move clockwise
        xc = -1.0 * self.model_states[:, 1]
        # Observe that np.angle(x + complex(0,1) * xc) == np.atan2(xc,x)
        # The y (in this case, xc) appears first in the atan2 formula
        return np.angle(x + complex(0, 1) * xc)
        # return np.atan2(xc,x)

    def cbt(self):
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
        Coupling strength parameter.
    gamma : float
        Amplitude relaxation parameter.
    beta : float
        Phase response parameter.
    a1 : float
        Amplitude of first-harmonic light effect.
    a2 : float
        Amplitude of second-harmonic light effect.
    betal1 : float
        Phase offset for first-harmonic light effect (radians).
    betal2 : float
        Phase offset for second-harmonic light effect (radians).
    sigma : float
        Baseline light-induced phase shift parameter.
    g : float
        Light sensitivity scaling parameter.
    alpha_0 : float
        Baseline light sensitivity parameter.
    delta : float
        Rate constant for adaptation variable n.
    p : float
        Power-law exponent for light input.
    i0 : float
        Saturation intensity for light sensitivity (lux).
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
        initial_condition=None
    ):
        if inputs is None or time is None:
            super().__init__(
                data=data,
                initial_conditions=np.array([0.82041911, 1.71383697, 0.52318122])
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
        # Integrate model and extract collective rhythm amplitude (r)
        return self.model_states[:, 0]

    def phase(self):
        # Integrate model and extract collective phase (phi)
        phi = self.model_states[:, 1]
        x = np.cos(phi)
        y = np.sin(phi)
        return np.angle(x + complex(0, 1) * y)

    def cbt(self):
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
    Implements the Hannay Two-Population (TP) model of human circadian rhythms [1]. Since the neuroanatomy of the
    suprachiasmatic nucleus (SCN) of the hypothalamus generally distinguishes between two populations, the ventral
    and dorsal, one with and the other without direct light input, Hannay et al. implemented a modified version of
    the Hannay Single Population module, attempting to better represent the neurophysiology of the central circadian
    pacemaker. Similar to the HannaySP, all three state variables are directly biologically interpretable. The model
    is also derived from the mathematical description of the rhythm within individual cells, from which a coherent
    behavior emerges.

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
        Intrinsic coupling strength within the ventral oscillator population.
    kdd : float
        Intrinsic coupling strength within the dorsal oscillator population.
    kvd : float
        Coupling strength from ventral to dorsal population.
    kdv : float
        Coupling strength from dorsal to ventral population.
    gamma : float
        Amplitude relaxation parameter.
    a1 : float
        Amplitude of first-harmonic light effect.
    a2 : float
        Amplitude of second-harmonic light effect.
    betal : float
        Phase offset for first-harmonic light effect (radians).
    betal2 : float
        Phase offset for second-harmonic light effect (radians).
    sigma : float
        Baseline light-induced phase shift parameter.
    g : float
        Light sensitivity scaling parameter.
    alpha_0 : float
        Baseline light sensitivity parameter.
    delta : float
        Rate constant for adaptation variable n.
    p : float
        Power-law exponent for light input.
    i0 : float
        Saturation intensity for light sensitivity (lux).
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
                initial_conditions=np.array([0.82423745, 0.82304996, 1.75233424, 1.863457, 0.52318122])
            )
        else:
            super().__init__(
                inputs=inputs,
                time=time,
                initial_conditions=np.array([0.82423745, 0.82304996, 1.75233424, 1.863457, 0.52318122]),
        )
        # Check for a scenario in which a initial condition is provided
        if initial_condition is not None:
            self.initial_conditions = initial_condition

        # self.initial_conditions = np.array([0.82423745, 0.82304996, 1.75233424, 1.863457, 0.52318122])
        # self.inputs = inputs
        # self.time = time
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
        # Integrate model and extract collective rhythm amplitude (r)
        return self.model_states[:, 0]

    def phase(self):
        # Integrate model and extract collective phase (phi)
        phi = self.model_states[:, 2]
        x = np.cos(phi)
        y = np.sin(phi)
        return np.angle(x + complex(0, 1) * y)

    def cbt(self):
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
    def __init__(
        self,
        time,
        inputs,
        window_size_days=4.0,
        esri_time_step_hours=1.0,
        initial_amplitude=0.1,
        midnight_phase=1.65238233,
    ):
        """
        Parameters:
        -----------
        :param inputs: The input light intensity levels over time
        :param time: The time points corresponding to the light levels
        :param window_size_days: The duration (in days) of the sliding window for ESRI calculation
        :param esri_time_step_hours: The step (in hours) for the movement of the sliding window for ESRI calculation
        :param initial_amplitude: initial amplitude for HannaySP model (it should be a low value)
        :param midnight_phase: Phase of the circadian model at midnight (in radians)
        """
        self.window_size = window_size_days
        self.esri_time_step = esri_time_step_hours
        self.initial_amplitude = initial_amplitude
        self.midnight_phase = midnight_phase

        self.time_vector = time
        self.light_vector = inputs

    def calculate(self):
        """
        Calculates the ESRI values over the specified range of time.

        Returns:
        --------
        esri_time : numpy.ndarray
            Array of time points at which ESRI is calculated.
        esri_values : numpy.ndarray
            ESRI values corresponding to esri_time.
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
        return esri_time, esri_values


class ModelComparer:
    def __init__(
        self,
        inputs,
        time,
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

        # Light and time vector
        self.time_vector = time
        self.light_vector = inputs

        # Instantiate forger model
        self.forger = Forger(inputs=self.light_vector, time=self.time_vector)
        if equilibrate:
            # First, calculate initial conditions based on the light and time vector
            ics = self.forger.get_initial_conditions(
                loop_number=50,
                light_vector=self.light_vector,
                time_vector=self.time_vector,
                change_params=True
            )

            # Second, integrate the equilibrated model and use last state as initial conditions
            new_ics = self.forger.integrate(
                light_vector=self.light_vector,
                time_vector=self.time_vector,
                initial_condition=ics
            )[-1]

            # Third, use the calculated initial conditions to define current model states
            self.forger.model_states = self.forger.integrate(
                light_vector=self.light_vector,
                time_vector=self.time_vector,
                initial_condition=ics
            )

        # Instantiate hannay model
        self.hannay = HannaySP(inputs=self.light_vector, time=self.time_vector)
        if equilibrate:
            # First, calculate initial conditions based on the light and time vector
            ics = self.hannay.get_initial_conditions(
                loop_number=50,
                light_vector=self.light_vector,
                time_vector=self.time_vector,
                change_params=True
            )

            # Second, integrate the equilibrated model and use last state as initial conditions
            new_ics = self.hannay.integrate(
                light_vector=self.light_vector,
                time_vector=self.time_vector,
                initial_condition=ics
            )[-1]

            # Third, use the calculated initial conditions to define current model states
            self.hannay.model_states = self.hannay.integrate(
                light_vector=self.light_vector,
                time_vector=self.time_vector,
                initial_condition=new_ics
            )

        # Store state variables of the forger model
        self.x = self.forger.model_states[:, 0]
        self.xc = self.forger.model_states[:, 1]

        # Store the collective phase of Hannay's model
        self.phase_vector = self.hannay.model_states[:, 1]

        # Predicted x and xc attributes
        self.predicted_x = None
        self.predicted_xc = None

        # Error of the prediction of x and xc
        self.error_x = None
        self.error_xc = None

    def predict_forger(self, change_params=False):
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
        # Attempt to fit a straight line (a polynomial of degree 1) to the collective phase
        coefficients = np.polyfit(self.time_vector, self.phase_vector, 1)
        slope, intercept = coefficients
        if change_params == True:
            self.phase_vector = slope * self.time_vector + intercept
        return slope * self.time_vector + intercept

    def find_optimal_params(self, change_params=False):
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

    def error(self, change_params=False):
        # Compute error associated with prediction
        error_x = self.x - self.predicted_x
        error_xc = self.xc - self.predicted_xc

        # Change the error_x and xc attributes
        if change_params:
            self.error_x = error_x
            self.error_xc = error_xc

        # Return the calculated vectors
        return error_x, error_xc

    def error_stats(self):
        def calculate_stats(prediction_error, measure):
            # Max and min error for the prediction
            max_error = prediction_error.max()
            min_error = prediction_error.min()

            # Calculate the error_band_width
            error_band_width = max_error - min_error

            # Calculate the range of values for the measure
            range_x = measure.max() - measure.min()

            # Calculate the magnitude of the error
            error_magnitude = (error_band_width / range_x) * 100

            # Return the calculated values
            return max_error, min_error, error_band_width, error_magnitude

        # Calculate x an xc error descriptive stats
        x_stats = calculate_stats(self.error_x, self.x)
        xc_stats = calculate_stats(self.error_xc, self.xc)
        return x_stats, xc_stats


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
    #dt = 1
    #dt=1/10 # for 10 bins/h
    time = np.arange(0, len(light) * dt, dt)


    # SECTION FOR COMPARING CIRCADIAN MODELS (FORGER AND HANNAY)
    comparison = ModelComparer(inputs=light, time=time, equilibrate=True)
    comparison.linearize_phase(change_params=True)
    comparison.find_optimal_params(change_params=True)
    comparison.predict_forger(change_params=True)
    comparison.error(change_params=True)
    max_error_x, min_error_x, error_band_x, magnitude_x = comparison.error_stats()[0]
    max_error_xc, min_error_xc, error_band_xc, magnitude_xc = comparison.error_stats()[1]

    # SECTION WITH GENERAL COMMANDS
    # hannay = HannaySP(inputs=light_vector, time=time_vector)
    # forger = HannaySP(inputs=light, time=time)
    # esri = ESRI(inputs=light_vector, time=time_vector)
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

    plt.figure()
    plt.plot(comparison.predicted_x, label="predicted_x", color="black", alpha=0.5)
    plt.plot(comparison.predicted_x, label="predicted_x", color="blue", alpha=0.9)
    #peaks,_= find_peaks(comparison.error_x)
    #for peak in peaks:
     #   print(f'{comparison.error_x[peak]:.3f}')
    plt.plot(comparison.error_x, label="Error", color="pink", alpha=0.9)
    plt.axhline(y=max_error_x, color="green", linestyle="--", label="Max Error")
    plt.axhline(y=min_error_x, color="purple", linestyle="--", label="Min error")
    plt.text(
        x=len(comparison.x) // 2,
        y=(max_error_x + min_error_x) / 2,
        s=f"Error band width {error_band_x:.2f}",
        color="orange",
        ha="center",
        va="center",
        fontsize=12,
        backgroundcolor="white",
    )
    plt.xlabel("Time")
    plt.ylabel("State variable")
    plt.title("State Evolution Over Time")
    plt.legend()
    plt.grid(True)
    #print(f"Magnitude of the error in predicting x = {magnitude_x}")

    #plt.figure()
    #plt.plot(comparison.predicted_xc, label="predicted_xc", color="black", alpha=0.5)
    #plt.plot(comparison.xc, label="observed_xc", color="blue", alpha=0.9)
    #plt.plot(comparison.error_xc, label="Error", color="pink", alpha=0.9)
    #plt.axhline(y=max_error_xc, color="green", linestyle="--", label="Max Error")
    #plt.axhline(y=min_error_xc, color="purple", linestyle="--", label="Min error")
    #plt.text(
     #   x=len(comparison.xc) // 2,
      #  y=(max_error_xc + min_error_xc) / 2,
       # s=f"Error band width {error_band_xc:.2f}",
        #color="orange",
        #ha="center",
        #va="center",
        #fontsize=12,
        #backgroundcolor="white",
    #)
    #plt.xlabel("Time")
    #plt.ylabel("State variable")
    #plt.title("State Evolution Over Time")
    #plt.legend()
    #plt.grid(True)
    #print(f"Magnitude of the error in predicting x = {magnitude_xc}")

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

    plt.show()

    # print(f"Execution time is {end-start} seconds.")


if __name__ == "__main__":
    main()
