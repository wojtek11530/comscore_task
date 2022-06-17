# ComScore task

Project contains solution to the ComScore application task.

All dependencies managed by poetry. To set up environment
```
poetry install
```

All models and data uploaded via DVC to Google Drive. Run
```
dvc pull
```
to obtain them.

# Method explanation

The model creation is done in notebook `notebooks/tfidf_classification.ipynb`.

The used method is not too fancy due to lack of time. It uses simple TF-IDF features
which dimension was further reduced to 2000 using SVD method. 

As a classifier is used XGBoost. The accuracy on validation set is 68% so regarding
equal balance of class means that is 18pp. better than random method.

Analysis from `notebooks/data_analysis.ipynb` shown that the data are cross-lingual. So using
of ready word embedding methods for a given language (e.g. fastText) can be a wrong way. But unfortuanettly
I did not manage to check it. Probably better resutls could be obtained with fastText model trianed on
training split and application of simple MLP using averaged word representation.

# Running

```commandline
python -m comscore_task.eval_model --input_filename [path_to_parquet_file] --output_filename [dir_with_output_parquet_file]
```

There is no handling of wrong paths in the script.
