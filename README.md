# predict_job_application
Report: `job_application_prediction_report.pdf`

### Steps to run the Jupyter Notebooks:
1. Create virtual environment: `python3 -m venv venv_job_pred &&  source venv_job_pred/bin/activate`
2. Install libraries using `pip install -r requirement.txt`.
3. Set the environment as the IPython kernel for Jupyter notebook execution: 
`ipython kernel install --user --name=venv_job_pred`
3. Add the dataset `job_desc.csv` and `user.csv` in `data/` directory.
5. Run the following  Notebooks (`data_exploring.ipynb` & `data_training.ipynb`) after initiating: `jupyter lab`

### Modules Description:
- `src.preprocessing_data.py` : Class used for data pre-processing: Data Cleaning and Feature Extraction.
- `src.util_training_evaluation.py`: Helper class used for splitting the data into train/ test set, training, and evaluating a given model on the test set.
- `src.util_evaluation_scores.py`: Helper method to compute evaluation scores for a given prediction.
- `data_exploring.ipynb`: Jupyter notebook used to explore the dataset and perform analysis to conlude on feature extraction and pre-processing steps for implementing the methods in `src.preprocessing_data.py`.
- `data_training.ipynb`: Jupyter notebook used to experiment with training different models and compare their evaluation results.# predict_job_application