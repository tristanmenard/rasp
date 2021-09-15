# rasp

`rasp` (**ra**dio **s**ignal **p**ropgation) is a tool for estimating VHF/UHF radio (10MHz ~ 1GHz) propagation losses over irregular terrain. Propagation losses are modelled according to the [Longley-Rice model](https://www.its.bldrdoc.gov/research-topics/radio-propagation-software/itm/itm.aspx). `rasp` is written in [Python 3](https://www.python.org), accelerated by [numba](http://numba.pydata.org) and [numpy](https://numpy.org/).

This version was altered by me to add compatibility with the TAFL Spectrum Management System database from ISED, extra functionality relating to saving transmitter and elevation data objects, as well as other minor tweaks. You can find the Gavin Noble's original version of `rasp` [here](https://gitlab.com/gnoble/rasp).
