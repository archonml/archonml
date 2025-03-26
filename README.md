# ArchOnML

[![DOI](10.26434/chemrxiv-2025-1g7jl)](https://doi.org/10.26434/chemrxiv-2025-1g7jl)

## About

<p>ArchOnML (derived from "Archive-On-Machine-Learning") is a python package that allows setting up and conducting machine learning-assisted molecular design
  projects. It presents an interface between quantum chemistry packages (e.g. *gaussian* and *orca*) and machine learning packages that requires only minimal user
  input to start your own machine learning project.</p>

<p>It is designed to be useful to beginners in both machine learning and quantum chemistry, while offering the flexibility to implement new, user-defined descriptors
   or machine learning models specific to the project requirements. If you choose to use ArchOnML for one of your projects, please cite the program package source from
   the github. (https://github.com/archonml/archonml) Further, there is a pre-print research article available that introduces ArchOnML on the example of anthraquinone
   available on ChemRxiv (https://chemrxiv.org/engage/chemrxiv/article-details/67d0668881d2151a0202d424).</p>

## Installation

### Requirements

<p>ArchOnML requires the installation of additional python packages. It has been developed using python version 3.7, so it is recommended to use a similar version. The package
   has been tested on UNIX and Windows machines and should be running on Macintosh systems as well, when embedded properly. It is highly recommended to use an anaconda environment,
   since a fresh installation may be easier to build than adding required packages to an existing python installation. After installing anaconda, you can start configuring a new
   python 3.7 environment by running</p>

> $ conda create -n AOML python=3.7 anaconda

<p>in a command line prompt. After entering the newly created environment, the further required packages to install are:</p>

- numpy
- pyquaternion
- mendeleev
- rdkit
- sklearn
- pickle
- matplotlib
- seaborn
- numba
- jupyter

<p>all of which are freely available via the PyPI repositories (i.e. by installing them via the *pip* command) and/or anaconda. Some of these packages may require installation from
   a non-standard source, such as conda-forge pr university hosted repositories. These can be specified, for example, by adding a **channel flag** (-c) as in</p>

> $ conda install -c conda-forge pyquaternion

<p>You can find these channels by searching for the packages on the https://anaconda.org/ website. Finally, after installing all packages successfully, it is
   recommended to run</p>

> $ conda update --all

<p>once, since some packages seem to consistently produce errors, otherwise.</p>

<p>Lastly, ArchOnML comes with many template IPython notebooks and python scripts that will help you set up a fresh project database in a step-by-step manner and monitor the
   learning process during model training in an interactive way. Here, the notebooks serve as an interim graphical user interface - so it is recommended to install and use jupyter
   notebooks for executing the package.</p>

### Instructions

<p>ArchOnML itself does not rquire a special installation or set-up. You may clone the latest package version from github to your desired target location via</p>

> $ git clone https://github.com/archonml/archonml.git

<p>Afterwards, to make the package availablke to your python environment, please add the path to the package (that is, the path to the folder **just above** the one that
   contains the init.py file inside of it) to the $PYTHONPATH environment variable of your preferred command prompt (bash, zshell, etc.). On Windows machines, the $PYTHONPATH
   variable can be added through the system settings via</p>

> My Computer \> Properties \> Advanced System Settings \> Environment Variables

<p>Note that the exact route may look different depending on your version of Windows (here, Windows 11).</p>

## Documentation

<p>A detailed documentation on the program package can be found in the repository at (https://github.com/archonml/archonml/blob/main/Documentation/ArchOnML_Documentation.pdf).
   This documentation contains a quick-start guide, general user's guide as well as a developer's guide - each explaining the code and its functionality in increasing level of
   detail.</p>

## Support and Contributions

<p>If you have trouble with installing or using the program package, please contact the developer of ArchOnML directly via github's
   issues tab (https://github.com/archonml/archonml/issues).</p>

## License Note

<p>ArchOnML is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation; either
   version 3 of the License, or (at your option) any later version.</p>

<p>The program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
   See the GNU Lesser General Public License for more details.</p>

<p>You should have received a copy of the GNU Lesser General Public License along with ArchOnML. If not, see https://www.gnu.org/home.en.html.</p>
