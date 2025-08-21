#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-10-13
# __author__ = Kshitij Kar, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = kar@gfz.de, https://github.com/Kshitij301199
# Please do not distribute this code without the author's permission

import os
# Set CUDA environment variables
os.environ["CUDA_HOME"] = "/storage/vast-gfz-hpc-01/cluster/nvidia/cuda/11.6.2"
os.environ["PATH"] = os.path.join(os.environ["CUDA_HOME"], "bin") + ":" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = os.path.join(os.environ["CUDA_HOME"], "lib64") + ":" + os.environ.get("LD_LIBRARY_PATH", "")
import argparse

import pandas as pd
import numpy as np

from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

from dataset2dataloader import *
from xlstm_model import *
from check_undetected_events import summary_results

from collections import Counter

def warmup_lambda(epoch):
    warmup_epochs = 3
    return min(1.0, (epoch + 1) / warmup_epochs)

def main(output_dir, input_station, model_type, feature_type, input_component, seq_length, batch_size):
    print(f"Start Job: UTC+0, {input_station, model_type, feature_type, input_component, seq_length, batch_size}",
          datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "\n")

    data_normalize = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    map_feature_size = {"A": 11, "B": 69, "C": 80}

    # load data
    input_data_year = [2017, 2018, 2019]
    input_features_name, X_train, y_train, time_stamps_train, _ = select_features(input_station, feature_type,
                                                                                  input_component, "training", input_data_year)
    input_data_year = [2020]
    input_features_name, X_test, y_test, time_stamps_test, _ = select_features(input_station, feature_type,
                                                                               input_component, "testing", input_data_year)
    print(f"Y-Label Test : {Counter(y_test)}")
    if data_normalize is True:
        X_train, X_test = input_data_normalize(X_train, X_test)
        X_train = pd.DataFrame(X_train, columns=input_features_name)
        X_test = pd.DataFrame(X_test, columns=input_features_name)
    else:
        pass

    # prepare the dataloader
    train_df = pd.concat([X_train, y_train, time_stamps_train], axis=1, ignore_index=True)
    train_sequences = data2seq(df=train_df, seq_length=seq_length)
    train_dataloader = dataset2dataloader(data_sequences=train_sequences, batch_size=batch_size,
                                          training_or_testing="training")

    test_df = pd.concat([X_test, y_test, time_stamps_test], axis=1, ignore_index=True)
    test_sequences = data2seq(df=test_df, seq_length=seq_length)
    test_dataloader = dataset2dataloader(data_sequences=test_sequences, batch_size=batch_size,
                                         training_or_testing="testing")

    # load model
    if model_type == 'xLSTM':
        train_model = xlstm_classifier(feature_size=map_feature_size.get(feature_type), device=device, hidden_size=256,
                                       num_blocks=2, slstm_at=[1], dropout=0.1, context_length=seq_length, num_layers=1)
    elif model_type == 'sLSTM':
        train_model = xlstm_classifier(feature_size=map_feature_size.get(feature_type), device=device, hidden_size=256,
                                       num_blocks=1, slstm_at=[0], dropout=0.1, context_length=seq_length, num_layers=1)
    elif model_type == 'mLSTM':
        train_model = xlstm_classifier(feature_size=map_feature_size.get(feature_type), device=device, hidden_size=256,
                                       num_blocks=1, slstm_at=[], dropout=0.1, context_length=seq_length, num_layers=1)
    summary(model=train_model,
            input_size=(batch_size, seq_length, map_feature_size.get(feature_type)),
            col_names=("input_size", "output_size", "num_params", "params_percent", "trainable"),
            device=device,
            depth=4)
    train_model.to(device)
    optimizer = torch.optim.Adam(train_model.parameters(), lr=0.0001)
    # Define scheduler: Reduce the LR by factor of 0.1 when the metric (like loss) stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3)
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    # warmup_scheduler = None

    # train and test
    trainer = xlstm_train_test(train_model,
                              optimizer,
                              train_dataloader.dataLoader(),
                              test_dataloader.dataLoader(),
                              device,
                              output_dir,
                              input_station,
                              model_type,
                              feature_type,
                              input_component,
                              scheduler,
                              warmup_scheduler)
    trainer.activation()

    # summary the results
    num_feats = map_feature_size.get(feature_type)
    summary_results(output_dir, input_station, model_type, feature_type, input_component, "training", num_feats)
    summary_results(output_dir, input_station, model_type, feature_type, input_component, "testing", num_feats)

    # vasulize the feature importance
    if feature_type == "C":
        pass
        # imp = shap_imp(input_station, model_type, feature_type, input_component, train_dataloader.dataLoader())

        # visualize_feature_imp("shap_value", imp, input_features_name,
        # input_station, model_type, feature_type, input_component)
    else:
        pass

    print(f"End Job: UTC+0, {input_station, model_type, feature_type, input_component, seq_length, batch_size}",
          datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "\n")


if __name__ == "__main__":
    # sinfo -n node[501-514] -N --Format="Nodelist,CPUsState,AllocMem,Memory,GresUsed,Gres"
    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument("--output_dir", default="path2model", type=str, help="check str path")
    parser.add_argument("--input_station", default="ILL12", type=str, help="input station")
    parser.add_argument("--model_type", default="Random_Forest", type=str, help="model type")
    parser.add_argument("--feature_type", default="C", type=str, help="feature type")
    parser.add_argument("--input_component", default="EHZ", type=str, help="seismic input_component")

    parser.add_argument("--seq_length", default=64, type=int, help="Input sequence length")
    parser.add_argument("--batch_size", default=16, type=int, help='Input batch size on each device')

    args = parser.parse_args()
    print(f"Arguments: {args}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Call the main function with parsed arguments
    main(args.output_dir, args.input_station, args.model_type, args.feature_type, args.input_component, args.seq_length, args.batch_size)
