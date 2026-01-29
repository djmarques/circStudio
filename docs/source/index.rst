.. circStudio documentation master file, created by
   sphinx-quickstart on Fri Jul 18 12:57:01 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

circStudio documentation
========================

`circStudio` is a Python package designed to combine mathematical models of circadian rhythms with
the analysis of actigraphy data. Actigraphy data are collected using a small actigraph unit (actimetry
sensor), which is worn over a given period of time, yielding time series of locomotor activity, light
exposure and skin temperature.

The development of `circStudio` was motivated by limitations in the supported Python versions of
`pyActigraphy` (tested for `python >= 3.7` and `python <= 3.9`) and by the need for a clearer
separation between data processing routines and functions for computing actigraphy-derived metrics.
In addition, although a Python package implementing several mathematical models of circadian rhythms
already exists (`circadian`), it supports input data only from a limited number of actimetry sensor
brands. Conversely, `pyActigraphy` supports data ingestion from a broad range o actimetry sensors,
but does not offer direct support for circadian rhythm modeling.

`circStudio` was therefore developed to bridge this gap by combining the data-handling and analysis
of `pyActigraphy` with established mathematical models of circadian rhythms originally implemented in
`circadian`. Furthermore, the numerical solver used to integrate the underlying systems of differential
equations was reimplemented to improve address improve numerical stability and robustness when applied
to actigraphy data collected in real-world settings.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorial_1
   tutorial_2
   tutorial_3
   tutorial_4
   api
