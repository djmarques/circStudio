**circStudio**
================
*Under construction*

**circStudio** is a Python package for mathematical modelling of circadian rhythms using actigraphy data. Built on top of the open-source package pyActigraphy, it retains key functionality from the original library while extending its capabilities. Specifically, the package enables the calculation of rest-activity rhythm metrics (e.g., interdaily stability [IS], intradaily variability [IV]), and light-derived measures (e.g., time above a given threshold [TAT], mean light timing [MLiT]).

In addition to the programmatic interface, **circStudio** will offer - in the next release - a graphical user interface, allowing users to perform actigraphy data analysis either interactively or through code. The motivation for this package stems from the need to analyze actigraphy data in Python 3.12 and later, while also integrating mathematical models of circadian rhythms into the analysis pipeline.


Citation
========

Citation of the original papers:

Hammad G, Reyt M, Beliy N, Baillet M, Deantoni M, Lesoinne A, et al. (2021) pyActigraphy: Open-source python package for actigraphy data visualization and    analysis. PLoS Comput Biol 17(10): e1009514. https://doi.org/10.1371/journal.pcbi.1009514

Hammad, G., Wulff, K., Skene, D. J., Münch, M., & Spitschan, M. (2024). Open-Source Python Module for the Analysis of Personalized Light Exposure Data from Wearable Light Loggers and Dosimeters. LEUKOS, 20(4), 380–389. https://doi.org/10.1080/15502724.2023.2296863

Tavella, F., Hannay, K., & Walch, O. (2023). Arcascope/circadian: Refactoring of readers and metrics modules, Zenodo, v1.0.2. https://doi.org/10.5281/zenodo.8206871

License
=======

This project keeps the same license as **pyActigraphy**, the the GNU GPL-3.0 License.


Acknowledgments
===============

Sincere thanks to the following teams:

* The developers of the original **pyActigraphy** package, whose work laid the foundation for this project (https://github.com/ghammad/pyActigraphy).
* The authors of the **circadian** package, whose original implementation of light-informed models was crucial for our implementation (https://github.com/Arcascope/circadian).