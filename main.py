#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import svm

col_header = ["time", "hand_type", "hand_id", "hand_confidence", "palm_x", "palm_y", "palm_z", "palm_width",
    "hand_x_basis_x", "hand_x_basis_y", "hand_x_basis_z", "hand_y_basis_x", "hand_y_basis_y", "hand_y_basis_z",
    "hand_z_basis_x", "hand_z_basis_y", "hand_z_basis_z",
    "thumb_len", "thumb_width", "thumb_direction_x", "thumb_direction_y", "thumb_direction_z",
    "thumb_velocity_x", "thumb_velocity_y", "thumb_velocity_z", "thumb_0_center_x", "thumb_0_center_y", "thumb_0_center_z",
    "thumb_1_center_x", "thumb_1_center_y", "thumb_1_center_z", "thumb_2_center_x", "thumb_2_center_y", "thumb_2_center_z",
    "thumb_3_center_x", "thumb_3_center_y", "thumb_3_center_z",
    "index_len", "index_width", "index_direction_x", "index_direction_y", "index_direction_z",
    "index_velocity_x", "index_velocity_y", "index_velocity_z", "index_0_center_x", "index_0_center_y", "index_0_center_z",
    "index_1_center_x", "index_1_center_y", "index_1_center_z", "index_2_center_x", "index_2_center_y", "index_2_center_z",
    "index_3_center_x", "index_3_center_y", "index_3_center_z",
    "middle_len", "middle_width", "middle_direction_x", "middle_direction_y", "middle_direction_z",
    "middle_velocity_x", "middle_velocity_y", "middle_velocity_z", "middle_0_center_x", "middle_0_center_y", "middle_0_center_z",
    "middle_1_center_x", "middle_1_center_y", "middle_1_center_z", "middle_2_center_x", "middle_2_center_y", "middle_2_center_z",
    "middle_3_center_x", "middle_3_center_y", "middle_3_center_z",
    "ring_len", "ring_width", "ring_direction_x", "ring_direction_y", "ring_direction_z", "ring_velocity_x", "ring_velocity_y",
    "ring_velocity_z", "ring_0_center_x", "ring_0_center_y", "ring_0_center_z", "ring_1_center_x", "ring_1_center_y", "ring_1_center_z",
    "ring_2_center_x", "ring_2_center_y", "ring_2_center_z", "ring_3_center_x", "ring_3_center_y", "ring_3_center_z",
    "pinky_len", "pinky_width", "pinky_direction_x", "pinky_direction_y", "pinky_direction_z",
    "pinky_velocity_x", "pinky_velocity_y", "pinky_velocity_z", "pinky_0_center_x", "pinky_0_center_y", "pinky_0_center_z",
    "pinky_1_center_x", "pinky_1_center_y", "pinky_1_center_z", "pinky_2_center_x", "pinky_2_center_y", "pinky_2_center_z",
    "pinky_3_center_x", "pinky_3_center_y", "pinky_3_center_z"]

def get_files(root_dir, extension = ".tsv"):
    """
    get all tsv files in root_dir
    fp_arr: file path
    bn_arr: basename path
    """
    fp_arr = []
    bn_arr = []
    for root, dirs, files in os.walk(root_dir):
        for fn in files:
            bn, ext = os.path.splitext(fn)
            if ext != extension:
                continue
            fp = os.path.join(root, fn)
            fp_arr.append(fp)
            bn_arr.append(bn)
    return fp_arr, bn_arr

def load_data(root_dir, col_header):
    fp_arr, bn_arr = get_files(root_dir)
    df_arr = []
    tg_arr = []
    for i in range(len(fp_arr)):
        fp = fp_arr[i]
        bn = bn_arr[i]
        df = pd.read_table(fp)
        df.columns = col_header
        num_row = df.shape[0]
        print("data shape: {} {}".format(fp, df.shape))
        df_arr.append(df)
        # basename is the target
        # create array with equal row of dataframe
        tgs = [bn for x in range(num_row)]
        tg_arr.extend(tgs)
    df_all = pd.concat(df_arr, sort=False)
    return df_all, tg_arr

def filter_col(df):
    # 0 (time), 1 (L, R) columns are not needed
    print("data shape: {}".format(df.shape))
    num_col = df.shape[1]
    return df.iloc[:,2:num_col]

def main():
    num_args = len(sys.argv)
    if num_args != 2:
        sys.exit("Insufficient args.\nusage) main.py path_to_data")
    data_path = sys.argv[1] # measuring rock, paper, or scissors

    df_all, tg = load_data(data_path, col_header)
    df = filter_col(df_all)
    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_val_score(clf, df, tg, cv=5)
    print(scores)

if __name__ == "__main__":
    main()
