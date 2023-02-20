 

The scripts and files in this repository are partly associted with the project presented with title 
"The integration of regional reflection seismic profiles and gravity datasets with different spatial coverage associated with geological models"


# To cite:

- [![DOI](https://zenodo.org/badge/603344068.svg)](https://zenodo.org/badge/latestdoi/603344068)


- Rashidifard, M., Giraud, J., Lindsay, M., Jessell, M. and Ogarko, V.: Constraining 3D geometric gravity inversion with 2D reflection seismic profile 
  using a generalized level-set approach: application to Eastern Yilgarn craton, Solid Earth Discuss., 1–35, doi:10.5194/se-2021-65, 2021.


- Mahtab Rashidifard. (2023). seismic and gravity input images for MVCNN. https://doi.org/10.5281/zenodo.7611550


- Mahtab Rashidifard, Jeremie Giraud, Mark Lindsay, & Mark Jessell. (2023). Cooperative geophysical 
  inversion integrated with 3D geological modelling. https://doi.org/10.5281/zenodo.7580824
  
- Mahtab Rashidifard, Jérémie Giraud, Mark Lindsay, Mark Jessell, & Vitaliy Ogarko. (2021). Constrained 3D geometric gravity inversion
  [Data set]. Zenodo. https://doi.org/10.5281/zenodo.4747913  







# README:
This file is a quick manual for using the provided codes and input files in this repository


# Seismic_forward:

The run scripts are provided for forward modelling of seismic traveltimes with three applications:

- Straight_ray_tracing.m: Used for well-tie tomography modelling

- Refracted_ray_tracing.m: Used for tracing refracted models (only for shallow structures)

- Full_waveform.m: Quick script that is used to model full-waveform of reflection seismic datasets 
(for more details refer too : Crewes, R., Grechka, V., & Wiggins, R. (2019). CREWES Matlab Toolbox. 
Retrieved from https://www.crewes.org/ResearchLinks/SoftwareLinks/CREWESMatlabToolbox))








# Synthetic_seismic: 

An alternative fast way to generate synthetic seismic datasets in time and depth domain

- synthetic_depth.py:  Quick way to generate seismic sections from initial AI model(in depth)

- Migration_depth_to_time.py: Image ray technique to transform the synthetic depth section to time domain

- RunScript_test.py: A sample script on using the functions








# Seismic_inversion: 

The script is provided for AI inversion along a seismic section in Boulia region

- AI_inversion.py: main script for AI inversion

- bresenham.py: Nothing but a function for extracting traces of seismics in 2D

- depth_to_time_utils.py: Functions for image-ray and modelling ray techniques






# Gravity:

Gravity.m: is the main script to run, which generate the model grid and datasets files of Yamarna Terrane in WA. This includes:
- model and data grid file for input to Tomofast-x inversion platform (refer to:"Ogarko, Vitaliy, Jeremie Giraud, and Roland Martin. "Tomofast-x v1. 0 source code." Zenodo (2021).")
  and "Giraud, J., Ogarko, V., Martin, R., Jessell, M. and Lindsay, M., 2021. Structural, petrophysical, and geological constraints in potential field inversion using the Tomofast-x v1.
  0 open-source code. Geoscientific Model Development, 14(11), pp.6681-6709."
  
- forward modelling of gravity datasets

- Extract grid and model along seismic profiles



# Deep_learning: 

- Autoencoder.py: Quick CNN model for recreating seismic images from time to depth
  Forward_Noddy: Used on Noddy generated models to fast forward seismic and gravity datasets 
  (Refer to https://essd.copernicus.org/articles/14/381/2022/ for more information about Noddyverse datasets)
  
- input_mvcnn.py: This script has been generated to save required input files for CNN purposes for both seismic and gravity datasets


- For the mvcnn code we refer to the https://github.com/jongchyisu/mvcnn_pytorch which has been utilised for training 
  Input datasets for training before processing are available and could be accessed from: https://zenodo.org/record/7611550#.Y-DgAXbP2Hs.
  The trained models are provided in the input datasets directory of this repository


- Two notebooks: 1- load-model-2022-11-18.ipynb and 2 - test_samples.ipynb are used for loading the trained models and testing the trained networks respectively. 

