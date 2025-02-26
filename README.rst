**pyActigraphy Lite (or Portable)**
================
*This is a work in progress* 
Open-source python package for actigraphy and light exposure data visualization and analysis. pyActigraphy Lite (or Portable) is intended to be a "easy-to-install" version of PyActigraphy, minimizing dependency conflicts. In order to achieve this goal, some trade-offs have been made. For instance, this portable version removes support for the accelerometer package (Axivity and Activinsights). It also does not include Numba support, with might make it might make matrix computations slower. However, it can be installed in recent versions of Python (up to Python 12.9 - currentyl working to make it compatible with later versions). 

Citation
========

Citation of the original paper:

  Hammad G, Reyt M, Beliy N, Baillet M, Deantoni M, Lesoinne A, et al. (2021) pyActigraphy: Open-source python package for actigraphy data visualization and    analysis. PLoS Comput Biol 17(10): e1009514. https://doi.org/10.1371/journal.pcbi.1009514

pyLight
=======

In the context of the Daylight Academy Project, `The role of daylight for humans <https://daylight.academy/projects/state-of-light-in-humans>`_ and
thanks to the support of its members, Dr. Mirjam Münch and Prof. `Manuel Spitschan <https://github.com/spitschan>`_,
a pyActigraphy module for analysing light exposure data has been developed, **pyLight**.
This module is part of the Human Light Exposure Database and is included in pyActigraphy version `v1.1 <https://github.com/ghammad/pyActigraphy/releases/tag/v1.1>`_ and higher.

A manuscript describing the *pyLight* module is available as a `preprint <https://osf.io/msk9n/>`_.

Code and documentation
======================

The pyActigraphy package is open-source and its source code is accessible `online <https://github.com/ghammad/pyActigraphy>`_.


An online documentation of the package is also available `here <https://ghammad.github.io/pyActigraphy/index.html>`_.
It contains `notebooks <https://ghammad.github.io/pyActigraphy/tutorials.html>`_ illustrating various functionalities of the package. Specific tutorials for the processing and the analysis of light exposure data with pyLight are also available.

Authors
=======

* **Grégory Hammad** `@ghammad <https://github.com/ghammad>`_ - *Initial and main developer*
* **Mathilde Reyt** `@ReytMathilde <https://github.com/ReytMathilde>`_

See also the list of `contributors <https://github.com/ghammad/pyActigraphy/contributors>`_ who participated in this project.

License
=======

This project is licensed under the GNU GPL-3.0 License - see the `LICENSE <LICENSE>`_ file for details

Acknowledgments
===============

* **Aubin Ardois** `@aardoi <https://github.com/aardoi>`_ developed the first version of the MTN class during his internship at the CRC, in May-August 2018.
* The CRC colleagues for their support, ideas, etc.
