**circStudio**
================

**circStudio** is a Python package for preprocessing, modeling, and analyzing actigraphy time series. It
enables users to read activity, light and temperature recordings collected by a wide range of actigraphy
devices, and provides conversion modules for commonly used systems (e.g., ActTrust, Actiwatch).

In addition to signal processing and common actigraphy-derived metrics, **circStudio** incorporates
mathematical models of circadian rhythms and algorithms for automatic sleep detection. This enables
users not only to characterize rest-activity patterns, but also to simulate circadian phase dynamics,
predicting sleep timing, and link actigraphy-derived signals to underlying physiological processes.

Core functionalities
--------------------
The main capabilities of **circStudio** are summarized below.
For complete documentation and examples, see `the online documentation <https://djmarques.github.io/circStudio/>`_.

Cleaning and preprocessing raw actigraphy data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Format-agnostic and flexible ``Raw`` class for importing actigraphy recordings
- Dedicated conversion modules for commonly used actigraphy file formats
- Automatic truncation of invalid or incomplete sequences at the beginning and/or end of recordings
- Detection of non-wear periods with optional imputation strategies for missing data

Common actigraphy-derived metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute standard activity- and light-derived metrics, including:

- **Interdaily Stability (IS)**
- **Intradaily Variability (IV)**
- **Rest–activity rhythm metrics**
- **Time Above Threshold (TAT)**
- **Mean Light Timing (MLiT)**

Mathematical models of circadian rhythms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A defining feature of ``circStudio`` is the inclusion of several mathematical models of
of circadian rhythms. Implemented models include:

- **Forger model**
- **Jewett model**
- **Hannay Single-Population (HannaySP)**
- **Hannay Two-Population (HannayTP)**
- **Hilaire 2007 model**
- **Skeldon 2023 model**
- **Breslow 2013 model** (melatonin dynamics)

These models enable users to:

- Predict circadian phase (Dim Light Melatonin Onset, DLMO) given a light schedule
- Model melatonin dynamics
- Infer sleep timing and circadian misalignment
- Integrate physiology-driven modeling with actigraphy-derived data

Design philosophy
-----------------

``circStudio`` unifies two complementary approaches to circadian research: data-driven actigraphy analysis and
mechanistic circadian modeling.

The package integrates preprocessing, rhythm quantification, and sleep detection capabilities from ``pyActigraphy``
with mathematical models of circadian dynamics provided by the ``circadian`` package.

By bridging actigraphy signal processing,  rhythm metrics, and physiology-based modeling, ``circStudio`` enables
researchers to move seamlessly from raw actigraphy recordings to predictions of circadian phase, sleep timing, and
circadian misalignment.

Citation
========

Citation of the original papers:

Hammad G, Reyt M, Beliy N, Baillet M, Deantoni M, Lesoinne A, et al. (2021) pyActigraphy: Open-source python package
for actigraphy data visualization and analysis. PLoS Comput Biol 17(10): e1009514.
https://doi.org/10.1371/journal.pcbi.1009514

Hammad, G., Wulff, K., Skene, D. J., Münch, M., & Spitschan, M. (2024). Open-Source Python Module for the Analysis of
Personalized Light Exposure Data from Wearable Light Loggers and Dosimeters.
LEUKOS, 20(4), 380–389. https://doi.org/10.1080/15502724.2023.2296863

Tavella, F., Hannay, K., & Walch, O. (2023). Arcascope/circadian: Refactoring of readers and metrics modules, Zenodo,
v1.0.2. https://doi.org/10.5281/zenodo.8206871

License
=======

This project keeps the same license as **pyActigraphy**, the GNU GPL-3.0 License.

Acknowledgments
===============

Sincere thanks to the following teams:

* The developers of the original **pyActigraphy** package, whose work laid the foundation for this project (https://github.com/ghammad/pyActigraphy).
* The authors of the **circadian** package, whose original implementation of light-informed models was crucial for our implementation (https://github.com/Arcascope/circadian).