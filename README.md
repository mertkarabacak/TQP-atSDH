# Machine Learning-Driven Prognostication in Traumatic Subdural Hematoma: Development of a Predictive Web Application

<i>A link to the paper discussing the details of the project will be provided upon acceptance.</i>

## Data

Restrictions apply to the availability of the data utilized in this study. Data were obtained from the American College of Surgeons (ACS)-Trauma Quality Program (TQP) and are available with the permission of the American College of Surgeons. If you want to replicate this study, you would need TQP-Participant Use Files (PUFs). 

Our preprocessing starts with a .csv file (isolated_atSDH.csv). This file was obtained through identifying isolated acute traumatic subdural hematoma patients in the main TQP-PUF, and merging various TQP-PUFs together.

## Repository Structure

The code provided in this repository is for preprocessing the data and testing machine learning algorithms for each outcome of interest in Jupyter Notebook format. 

*TQP_atSDH_Preprocess.ipynb
*TQP_atSDH_Mortality.ipynb
*TQP_atSDH_Discharge.ipynb
*TQP_atSDH_LOS.ipynb
*TQP_atSDH_ICU-LOS.ipynb
*TQP_atSDH_Complications.ipynb

The file app.py is the source code of the web application deployed in Hugging Face (https://huggingface.co/spaces/MSHS-Neurosurgery-Research/TQP-atSDH).

*app.py

## Requirements

*numpy
*pandas
*matplotlib
*math
*scipy
*random
*sklearn
*optuna
*tabpfn
*pytorch-tabnet
*shap

## Contact

Mert Karabacak, MD
Clinical Research Coordinator
Mount Sinai Health System
Department of Neurosurgery

Email: Mert.Karabacak@mountsinai.org
X: @MertKarabacakMD
