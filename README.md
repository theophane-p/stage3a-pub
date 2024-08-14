# 2024 Research Internship Public Code

This repository contains code written during my École polytechnique 3rd year resarch internship with specialization MAP594 - *Modélisation probabiliste et statistique* (probabilistic and statistical modelling). This internship took place under the supervision of Jo Eidsvik at the Center for Geophysical Forecasting (CGF) from the Norwegian University of Science and Technology (NTNU, *Norges teknisk-naturvitenskapelige universitet*).

Since the Distributed Acoustic Sensing (DAS) data I used is not open-access, this code is not enough to reproduce the results exposed in the report. However, it can help in understanding how I came to these results.

## License

This project is licensed under the GNU Lesser General Public License version 3 (LGPLv3). See the [LICENSE](LICENSE) file for details.

## Reproducing part of the results

### Python environment

The Python environment I use is generated using [mamba](https://mamba.readthedocs.io/en/latest/index.html) and [mamba specification file](env/stb.yml) in order to create an isolated python version, and dependencies are then installed with [pip](https://pypi.org/project/pip/) and a [file listing dependencies](env/requirements.txt).

The [mambafreeze.txt](env/mambafreeze.txt) and [pipfreeze.txt](env/pipfreeze.txt) might help in having a working environment.

### paths

Many files require `data.paths`. You can rename (and modify) [this file](src/data/paths_todo.py) in order to make them work properly.

### Downloading data

- Weather station data is available on [seklima.no](https://seklima.met.no/)
- Weather analysis data can be downloaded and processed with files in the [weather_analysis](src/data/weather_analysis/) folder
- OpenStreetMap data on the power line can be downloaded on [overpass-turbo.eu](https://overpass-turbo.eu/) with the query below and processed with [this file](src/report_plots/geography_processing.py)

```
way
  [name="Fygle - Solbjørn"]
  ({{bbox}});
(._;>;);
out;
```

### Running code

The python scripts are meant to be run at the root of this directory. You may have to modify relative paths otherwise. 
