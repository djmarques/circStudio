def _lids_func(x):
    """LIDS transformation function

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
    return 100/(x+1)


def _lids_inverse_func(x):
    """LIDS inverse transformation function

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
    return 100/x - 1