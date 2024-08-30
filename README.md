# Active-Learning-Using-various-representations

# This repository is for the active learning strategy implemented on various data representations in organic photovoltaics.

## Data description
The data is in 'Data' folder, which contains input features and output property Jsc

## Usage
- Run main.py by selecting different active learning algorithms
- There are 2 choices for getting input data: `get_input` for Active learning with known salient features, `get_input_all` for Active learning without known salient features
- After run main.py, all results will store in the folder: 'Results_Data' or 'Results_Plot'
## Analysis results
- The results are in the folder: 'Results_Data' or 'Results_Plot'

## Requirements for libraries
- numpy
- matplotlib
- pandas
- scikit-learn
- scipy
- pot

## Citation

If you use this repository in your research, please cite it as follows:

```
@Article{D4DD00073K,
author ="Liu, Hao and Yucel, Berkay and Ganapathysubramanian, Baskar and Kalidindi, Surya R. and Wheeler, Daniel and Wodo, Olga",
title  ="Active learning for regression of structure–property mapping: the importance of sampling and representation",
journal  ="Digital Discovery",
year  ="2024",
pages  ="-",
publisher  ="RSC",
doi  ="10.1039/D4DD00073K",
url  ="http://dx.doi.org/10.1039/D4DD00073K",
abstract  ="Data-driven approaches now allow for systematic mappings from materials microstructures to materials properties. In particular{,} diverse data-driven approaches are available to establish mappings using varied microstructure representations{,} each posing different demands on the resources required to calibrate machine learning models. In this work{,} using active learning regression and iteratively increasing the data pool{,} three questions are explored: (a) what is the minimal subset of data required to train a predictive structure–property model with sufficient accuracy? (b) Is this minimal subset highly dependent on the sampling strategy managing the datapool? And (c) what is the cost associated with the model calibration? Using case studies with different types of microstructure (composite vs. spinodal){,} dimensionality (two- and three-dimensional){,} and properties (elastic and electronic){,} we explore these questions using two separate microstructure representations: graph-based descriptors derived from a graph representation of the microstructure and two-point correlation functions. This work demonstrates that as few as 5% of evaluations are required to calibrate robust data-driven structure–property maps when selections are made from a library of diverse microstructures. The findings show that both representations (graph-based descriptors and two-point correlation functions) can be effective with only a small quantity of property evaluations when combined with different active learning strategies. However{,} the dimensionality of the latent space differs substantially depending on the microstructure representation and active learning strategy."}
```
Or
```
@software{hliu_OPV2D_for_Active_2024,
   author = {Liu, Hao},
   doi = {10.5281/zenodo.12701801},
   month = jul,
   title = {{Active Learning Using Various Representations OPV}},
   url = {https://github.com/hliu56/Active-Learning-Using-various-representations},
   version = {0.1},
   year = {2024}
}
```

## Licensing, Authors, and Acknowledgements
This repository is on Apache License 2.0 license.


