## Matrix Factorization with Landmarks for Spatial Data

### File Structure
+ `code/`: Source code of SMFL in Python, `evaluation.py` is the entry of the experiment.
+ `code/datasets_missing`: All datasets with masks used in the experiments.
+ `data/`: The raw data of the datasets used in the paper, details of the datasets as well as pre-processing could be found in the papr. 
  + Since Vehicle Dataset is collected by our industrial partner, we do not include the raw data here, but the desensitized and masked experimental data are still provided in `code/datasets_missing`.

+ `docs/`: Appendix of the paper


### Example Invocation
requirement:
> scikit-learn = 1.0.2
> pandas = 1.4.2

example invocation:
> python evaluation.py





