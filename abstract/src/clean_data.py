#/usr/bin/env python

from os.path import join, split
from gaitanalysis import motek, gait

# TODO : Load this path from a configuration file, our download from
# Figshare once I post the data.
root_data_directory = "/home/moorepants/Data/human-gait/gait-control-identification"

mocap_file_path = join(root_data_directory, 'T009', 'mocap-009.txt')
record_file_path = join(root_data_directory, 'T009', 'record-009.txt')
meta_file_path = join(root_data_directory, 'T009', 'meta-009.yml')

dflow_data = motek.DFlowData(mocap_file_path, record_file_path,
                             meta_file_path)
dflow_data.clean_data(interpolate_markers=True)

# 'TreadmillPerturbation' is the current name of the longitudinal
# perturbation trials. This returns a data frame of processed data.
perturbation_data_frame = \
    dflow_data.extract_processed_data(event='Longitudinal Perturbation',
                                      index_col='TimeStamp')

# Here I compute the joint angles, rates, and torques.
inv_dyn_low_pass_cutoff = 6.0  # Hz
inv_dyn_labels = motek.markers_for_2D_inverse_dynamics()


def add_negative_columns(data):
    """Creates new columns in the DataFrame for any D-Flow measurements in
    the Z axis."""
    new_inv_dyn_labels = []
    for label_set in inv_dyn_labels:
        new_label_set = []
        for label in label_set:
            if 'Z' in label:
                new_label = 'Negative' + label
                data[new_label] = -data[label]
            else:
                new_label = label
            new_label_set.append(new_label)
        new_inv_dyn_labels.append(new_label_set)
    return new_inv_dyn_labels


new_inv_dyn_labels = add_negative_columns(perturbation_data_frame)

perturbation_data = gait.WalkingData(perturbation_data_frame)

#args = new_inv_dyn_labels + [dflow_data.meta['subject']['mass'],
                             #inv_dyn_low_pass_cutoff]
args = new_inv_dyn_labels + [79.0, inv_dyn_low_pass_cutoff]

perturbation_data.inverse_dynamics_2d(*args)

# The following identifies the steps based on vertical ground reaction
# forces.
perturbation_data.grf_landmarks('FP2.ForY', 'FP1.ForY',
                                filter_frequency=15.0,
                                num_steps_to_plot=None, do_plot=False,
                                threshold=30.0, min_time=290.0)

perturbation_data.split_at('right', num_samples=20,
                           belt_speed_column='RightBeltSpeed')

perturbation_data.save(join(split(__file__)[0], '../data/perturbation.h5'))
