import numpy as np
import pandas as pd


def _window_convolution(x, scale, window):
    """
    References
    ----------
    This code is derived from the original implementation in pyActigraphy, distributed under the BSD 3-Clause License.
    Original author: Grégory Hammad (gregory.hammad@uliege.be).

    [1] Hammad, G., Reyt, M., Beliy, N., Baillet, M., Deantoni, M., Lesoinne, A., Muto, V., & Schmidt, C. (2021).
    pyActigraphy: Open-source python package for actigraphy data visualization and analysis.
    PLoS Computational Biology, 17(10), 1009514–1009535. https://doi.org/10.1371/journal.pcbi.1009514

    [2] Hammad, G., Wulff, K., Skene, D. J., Münch, M., & Spitschan, M. (2024). Open-Source Python Module for the
    Analysis of Personalized Light Exposure Data from Wearable Light Loggers and Dosimeters.
    LEUKOS, 20(4), 380–389. https://doi.org/10.1080/15502724.2023.2296863
    """
    return scale * np.dot(x, window)


def _calculate_score(activity, wa, wp, p):
    """
    References
    ----------
    This code is derived from the original implementation in pyActigraphy, distributed under the BSD 3-Clause License.
    Original author: Grégory Hammad (gregory.hammad@uliege.be).

    [1] Hammad, G., Reyt, M., Beliy, N., Baillet, M., Deantoni, M., Lesoinne, A., Muto, V., & Schmidt, C. (2021).
    pyActigraphy: Open-source python package for actigraphy data visualization and analysis.
    PLoS Computational Biology, 17(10), 1009514–1009535. https://doi.org/10.1371/journal.pcbi.1009514

    [2] Hammad, G., Wulff, K., Skene, D. J., Münch, M., & Spitschan, M. (2024). Open-Source Python Module for the
    Analysis of Personalized Light Exposure Data from Wearable Light Loggers and Dosimeters.
    LEUKOS, 20(4), 380–389. https://doi.org/10.1080/15502724.2023.2296863
    """
    weigths = np.concatenate([wa, wp])

    scores = activity.rolling(len(weigths), center=True).apply(
        _window_convolution, args=(p, weigths), raw=True
    )

    return scores


def _calculate_state(
    data,
    istate,
    wa,
    wp,
    p,
    positive_rescoring,
    negative_rescoring,
    positive_state
):

    """Sleep (re)scoring rules for the Condor Sleep Model

    These rules are applied to a series of initial states (wake, rest or sleep)
    and modify them as a function of their scoring.

    Parameters
    ----------
    data : pandas.Series
        Input activity data.
    istate: array_like
        Array of initial states.
    wa: array_like
        Array of weights for the epochs before the one being evaluated and for
        the current epoch (current epoch is the last index of the array).
    wp: array_like
        Array of weights for the epochs after the one being evaluated.
    p: float
        Overall scaling for the weighted moving average used for the scoring.
    positive_rescoring: int
        Number of consecutive indexes of the score array that need to be below
        1 to change the state to the "positive" state.
    negative_rescoring: int
        Number of consecutive indexes of the score array that need to be above
        1 to keep the initial state.
    positive_state: int
        Index of the state that an initial state is changed to when it is
        rescored.

    Returns
    -------
    state : array_like
        Array of states.

    References
    ----------
    This code is derived from the original implementation in pyActigraphy, distributed under the BSD 3-Clause License.
    Original author: Grégory Hammad (gregory.hammad@uliege.be).

    [1] Hammad, G., Reyt, M., Beliy, N., Baillet, M., Deantoni, M., Lesoinne, A., Muto, V., & Schmidt, C. (2021).
    pyActigraphy: Open-source python package for actigraphy data visualization and analysis.
    PLoS Computational Biology, 17(10), 1009514–1009535. https://doi.org/10.1371/journal.pcbi.1009514

    [2] Hammad, G., Wulff, K., Skene, D. J., Münch, M., & Spitschan, M. (2024). Open-Source Python Module for the
    Analysis of Personalized Light Exposure Data from Wearable Light Loggers and Dosimeters.
    LEUKOS, 20(4), 380–389. https://doi.org/10.1080/15502724.2023.2296863
    """
    # create a local copy of the state
    state = istate.copy()

    # calculate the score
    scores = _calculate_score(data, wa, wp, p)

    # replace NaN produced by the convolution with default scores
    scores.iloc[:len(wa)-1] = 0
    scores.iloc[-len(wp):] = 0

    # apply rescoring
    is_positive_state = False
    for i in range(len(wa)-1, (len(scores)-len(wp))):
        # negative rescoring:
        # if score is lower than 1 (positive state), check if
        if is_positive_state:
            if scores[i] < 1:
                state[i] = positive_state
            else:
                is_positive_state = False
                for subindex in range(i+1, i+negative_rescoring+1):
                    if scores[subindex] < 1:
                        state[i] = positive_state
                        is_positive_state = True
                        break
        else:
            if scores[i] < 1:
                is_positive_state = True
                for subindex in range(i+1, i+positive_rescoring+1):
                    if scores[subindex] >= 1:
                        is_positive_state = False
                        break
            if is_positive_state:
                state[i] = positive_state

    for i in range(0, len(wa)-1):
        state[i] = state[len(wa)-1]

    for i in range(len(scores)-len(wp), len(scores)):
        state[i] = state[len(scores)-len(wp)-1]

    return state


def csm(
    data,
    wa=np.array([34.5, 133, 529, 375, 408, 400.5, 1074, 2048.5, 2424.5]),
    wp=np.array([1920, 149.5, 257.5, 125, 111.5, 120, 69, 40.5]),
    p_rest=0.00005,
    p_sleep=0.000464,
    pr_rest=0,
    nr_rest=0,
    pr_sleep=1,
    nr_sleep=0,
    score_rest=2,
    score_sleep=1
):
    """Condor Sleep Model

    Sleep-wake scoring algorithm developed by Condor Instrument for their
    ActTrust devices.

    This algorithm works in a two-step fashion. First, it classifies all epochs
    as wake or rest, as function of each epoch's score. Second, using a more
    stringent scoring threshold, "rest" epoch are re-classified as "sleep". A
    relabelling mechanism using the labels of the surrounding epochs is also
    applied to consolidate periods of epochs labelled as rest or sleep.

    Parameters
    ----------
    data : pandas.Series
        Input activity data.
    istate: array_like
        Array of initial states.
    wa: array_like
        Array of weights for the epochs before the one being evaluated and for
        the current epoch (current epoch is the last index of the array).
    wp: array_like
        Array of weights for the epochs after the one being evaluated.
    p_rest: float
        Overall scaling for the weighted moving average used for the scoring
        during the detection of "rest" epochs.
    p_sleep: float
        Overall scaling for the weighted moving average used for the scoring
        during the detection of "sleep" epochs.
    pr_rest: int
        Number of consecutive indexes of the score array that need to be below
        1 to change the state to the "rest" state.
    nr_rest: int
        Number of consecutive indexes of the score array that need to be above
        1 to keep the initial state during the detection of "rest" epochs.
    pr_sleep: int
        Number of consecutive indexes of the score array that need to be below
        1 to change the state to the "sleep" state.
    nr_sleep: int
        Number of consecutive indexes of the score array that need to be above
        1 to keep the initial state during the detection of "sleep" epochs.
    score_rest: int
        State index for epochs labelled as "rest".
    score_sleep: int
        State index for epochs labelled as "sleep".

    Returns
    -------
    csm : pandas.Series
        Series of state indices.

    References
    ----------
    This code is derived from the original implementation in pyActigraphy, distributed under the BSD 3-Clause License.
    Original author: Grégory Hammad (gregory.hammad@uliege.be).

    [1] Hammad, G., Reyt, M., Beliy, N., Baillet, M., Deantoni, M., Lesoinne, A., Muto, V., & Schmidt, C. (2021).
    pyActigraphy: Open-source python package for actigraphy data visualization and analysis.
    PLoS Computational Biology, 17(10), 1009514–1009535. https://doi.org/10.1371/journal.pcbi.1009514

    [2] Hammad, G., Wulff, K., Skene, D. J., Münch, M., & Spitschan, M. (2024). Open-Source Python Module for the
    Analysis of Personalized Light Exposure Data from Wearable Light Loggers and Dosimeters.
    LEUKOS, 20(4), 380–389. https://doi.org/10.1080/15502724.2023.2296863
    """

    # start with all states awake
    state0 = np.zeros(len(data))

    stateR = _calculate_state(
        data, state0, wa, wp, p_rest, pr_rest, nr_rest, score_rest
    )
    stateF = _calculate_state(
        data, stateR, wa, wp, p_sleep, pr_sleep, nr_sleep, score_sleep
    )

    return pd.Series(index=data.index, data=stateF)
