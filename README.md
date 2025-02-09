# Prediction of properties of a thermoelectric materials using Machine Learning


## Motivation
There is a one to one correspondence between technological advancements and material innovation. One such hot topic of interest is thermoelectic materials. The applications avary over a wide range liek eletricity generation, refrigeration, air condition, particular heating/cooling, biomedical devices etc. Machine learning is a talk of the town for at least a decade now. It is also strongeest tool for the prediction of new age materials with desirable properties. The possiblities are endless and potential is huge.


## Primary Calculations
Implementation of the following paper :
- Prediction of Seebeck Coefficient for Compounds without Restriction to Fixed Stoichiometry A Machine Learning Approach by Al’ona Furmanchuk et. al.

## Current work
Various properties of 24,759 bulk and 2D materials computed with the OptB88vdW and TBmBJ functionals taken from the JARVIS DFT database.

- Featurization is carried out using compostion and structural information given in database.
– After preprocessing we have implemented random forest regressor and carried out cross validation, evaluation of the accuracy of the model.
