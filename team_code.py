#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from helper_code import *
from features_extractor import *

import random
import pandas as pd

from imblearn.over_sampling import SMOTE
from collections import Counter

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

def train_model(data_folder, model_folder, verbose):
    if verbose:
        print('Finding the Challenge data...')

    # Load all records from the data folder
    records = find_records(data_folder)
    records, num_records = select_records(data_folder, records)
    print('Number of samples:', num_records)
    
    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')
    
    features = []
    labels = []

    # Iterate over the records.
    error = 0
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])

        feature = extract_features(record)

        numeric_feature = np.array(feature, dtype=float)
        
        if np.isnan(numeric_feature).any():
            error += 1
            continue

        features.append(numeric_feature)
        labels.append(load_label(record))
    
    features = np.stack(features)
    labels = np.stack(labels)
    if verbose:
        print("Signal error counter:", error)

    # ----------------------------
    # Conteo antes de SMOTE
    # ----------------------------
    label_counts_before = Counter(labels)
    print("Class distribution BEFORE SMOTE:")
    for cls, count in label_counts_before.items():
        label_str = "Positive" if cls else "Negative"
        print(f"{label_str}: {count}")

    # ----------------------------
    # Aplicar SMOTE
    # ----------------------------
    smote = SMOTE()
    features, labels = smote.fit_resample(features, labels)

    # ----------------------------
    # Conteo después de SMOTE
    # ----------------------------
    label_counts_after = Counter(labels)
    print("Class distribution AFTER SMOTE:")
    for cls, count in label_counts_after.items():
        label_str = "Positive" if cls else "Negative"
        print(f"{label_str}: {count}")

    # ----------------------------
    # Entrenamiento del modelo
    # ----------------------------
    if verbose:
        print('Training the model on the data...')

    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'eval_metric': ['logloss']
    }

    model = XGBClassifier(eval_metric='logloss')
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               scoring='f1_weighted', cv=5, n_jobs=-1)

    grid_search.fit(features, labels)
    best_model = grid_search.best_estimator_

    os.makedirs(model_folder, exist_ok=True)
    save_model(model_folder, best_model)

    if verbose:
        print('Training complete.')
        
        

def process_record(record_path):
    try:
        feature = extract_features(record_path)
        if feature is None or np.isnan(feature).any():
            return None, None
        label = load_label(record_path)
        return feature, label
    except Exception as e:
        print(f"Error procesando {record_path}: {e}")
        return None, None


# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'model.sav')
    model = joblib.load(model_filename)
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    # Load the model.
    model = model['model']

    # Extract the features.
    features = extract_features(record)
    numeric_feature = np.array(features, dtype=float)
    if np.isnan(numeric_feature).any():
        return 0, 0
    
    features = features.reshape(1, -1)

    # Get the model outputs.
    binary_output = model.predict(features)[0]
    probability_output = model.predict_proba(features)[0][1]

    return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

def extract_features(record):
    try:
        header = load_header(record)
        age = get_age(header)
        sex = get_sex(header)
        sfreq = int(get_sampling_frequency(header))

        one_hot_encoding_sex = np.zeros(3, dtype=bool)
        if sex == 'Female':
            one_hot_encoding_sex[0] = 1
        elif sex == 'Male':
            one_hot_encoding_sex[1] = 1
        else:
            one_hot_encoding_sex[2] = 1

        signal, fields = load_signals(record)

        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(signal)

            # Extraer características ECG base
        base_features = extract_ecg_features(normalized_data, channel=1, fs=sfreq)

        if base_features is None:
            return None

        # Agregar características adicionales
        base_features['frequency_domain'] = frequency_domain_analysis(normalized_data[:, 1], sfreq)
        #base_features['nonlinear'] = nonlinear_features(normalized_data[:, 1])
        base_features['signal_characteristics'] = additional_signal_characteristics(normalized_data[:, 1])
        base_features['wavelet_features'] = complex_wavelet_transform(normalized_data[:, 1])

        # Agregar información demográfica
        base_features['demographic'] = {
            'age': age,
            'sex_encoding': one_hot_encoding_sex
        } 

        features = flatten_features_dict(base_features)
        if any(is_invalid(f) for f in features):
            return None

        return features

    except Exception as e:
        print(f"Error al procesar el registro {record}: {e}")
        return None
    

# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)
    
    
    
#####################################################################################
def is_invalid(x):
    if x is None:
        return True
    try:
        return not np.isfinite(float(x))
    except (ValueError, TypeError):
        return True

def filter_records_by_folder(records, folder_keyword):
    """Return all records that contain a specific folder in their path."""
    return [f for f in records if folder_keyword in os.path.normpath(os.path.dirname(f))]

def sample_records(records, max_samples):
    """Return a random subset of records, up to max_samples."""
    return random.sample(records, min(max_samples, len(records)))

def select_records(data_folder, records, max_ptb = 3000, max_negative_code = 6000, max_positive_code = 6000):
    # Identify PTB-XL records and remove them from the list
    PTB_records = filter_records_by_folder(records, "PTB-XL")
    records = list(set(records) - set(PTB_records))
    PTB_records = sample_records(PTB_records, max_ptb)

    # Identify CODE-15% records and remove them from the list
    code15_records = [f for f in records if os.path.dirname(f).endswith("CODE-15%")]
    records = list(set(records) - set(code15_records))
    
    positives, negatives = [], []
    #print(code15_records)
    for rc in code15_records:
        rc_path = os.path.join(data_folder, rc)
        label_rc = load_label(rc_path)
        #print(label_rc)
        if(label_rc == True):
            positives.append(rc)
        elif(label_rc == False):
            negatives.append(rc)
        else:
            continue;      
    
    positives = sample_records(positives, max_positive_code)
    negatives = sample_records(negatives, max_negative_code)
    # Combine all selected records
    final_records = records + PTB_records + positives + negatives
    return final_records, len(final_records)
