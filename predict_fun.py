import numpy as np
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from datasets import ECGSequence


def predict_diagnosis(path_to_hdf5, path_to_model,
                      dataset_name='tracings',
                      output_csv_file="./dnn_output.csv",
                      bs=32):

    # Import data
    seq = ECGSequence(path_to_hdf5, dataset_name, batch_size=bs)
    # Import model
    model = load_model(path_to_model, compile=False)
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    y_score = model.predict(seq,  verbose=1)

    ############################################################################
    # PierMOD: Part of the original predict.py script customized to allow usage
    # for Health Big Data project purposes

    # Additional imports
    import pandas as pd

    # Create dataframe with prediction scores from numpy array (one row for
    # each ECG and one column for each diagnosis)
    diagnosis = np.array(['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST'])
    df = pd.DataFrame(y_score, columns=diagnosis)

    # THRESHOLDS: Found in the script "generate_figures_and_tables.py":
    # Get threshold that yield the best precision recall using "get_optimal_precision_recall" on validation set
    # (we rounded it up to three decimal cases to make it easier to read...)
    threshold = np.array([0.124, 0.07, 0.05, 0.278, 0.390, 0.174])
    mask = y_score > threshold

    # Generate prediction labels by comparing scores with thresholds
    y_labels = []
    for r in range(mask.shape[0]):
        if np.any(mask[r,:]):
            joinedLabels = '_'.join(diagnosis[ np.asarray(mask[r,:]).nonzero() ])
            y_labels.append(joinedLabels)
        else:
            y_labels.append('NoAbnormalities')
        
    # Adding prediction labels to the dataframe
    df['PredictedLabels'] = y_labels

    # Save dataframe to CSV file
    df.to_csv(output_csv_file, index=False, sep=',')

    print("CSV output with prediction scores and labels saved")
