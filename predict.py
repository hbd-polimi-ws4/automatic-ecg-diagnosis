import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from datasets import ECGSequence


if __name__ == '__main__':

    ##################################################################################################
    # Choose one of the following options by commenting the other one.
    #---OPTION 1 (original by Antonior92): For production use through terminal
    parser = argparse.ArgumentParser(description='Get performance on test set from hdf5')
    parser.add_argument('path_to_hdf5', type=str,
                        help='path to hdf5 file containing tracings')
    parser.add_argument('path_to_model',  # or model_date_order.hdf5
                        help='file containing training model.')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='name of the hdf5 dataset containing tracings')
    parser.add_argument('--output_file', default="./dnn_output.npy",  # or predictions_date_order.csv
                        help='output npy file.')
    parser.add_argument('-bs', type=int, default=32,
                        help='Batch size.')

    args, unk = parser.parse_known_args()
    if unk:
        warnings.warn("Unknown arguments:" + str(unk) + ".")
    
    #---OPTION 2 (PierMOD): To debug this script through VSCode directly, with the model trained by
    #                       Ribeiro et al. (2020) and the test dataset they provide publicly. See
    #                       README.md to download those.
    # class script_args:
    #     path_to_hdf5='./test_data/ecg_tracings.hdf5'
    #     path_to_model='./model/model.hdf5'
    #     dataset_name='tracings'
    #     output_file="./dnn_output.npy"
    #     bs=32
    # args = script_args()

    ##################################################################################################


    # Import data
    seq = ECGSequence(args.path_to_hdf5, args.dataset_name, batch_size=args.bs)
    # Import model
    model = load_model(args.path_to_model, compile=False)
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    y_score = model.predict(seq,  verbose=1)

    # Save prediction scores array to .npy file
    np.save(args.output_file, y_score)

    print("Output predictions saved")

    ############################################################################
    # PierMOD: Part customized to allow usage for Health Big Data project
    # purposes

    # Additional imports
    import pandas as pd
    from pathlib import Path

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
    ext = Path(args.output_file).suffix
    if ext == '':
        output_csv_file = args.output_file + '.csv'
    else:
        output_csv_file = args.output_file.replace(ext, '.csv')
    df.to_csv(output_csv_file, index=False, sep=',')
