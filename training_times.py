import pandas as pd
import numpy as np
import os

ROOT_FOLDER = './pretrained/hopper-medium-v2/retnet/pretrained/2023-11-29_19-15-42'
df = pd.read_csv("pretrained/hopper-medium-v2/retnet/pretrained/2023-11-29_19-15-42/time_table.csv")

time_train = df["time_trainer (s)"]
time_plan = df["time_epoch (s)"] - df["time_trainer (s)"]

print("Average time for training: ", np.mean(time_train)/60)
print("Standard deviation for training: ", np.std(time_train)/60)
print("Average time for planning: ", np.mean(time_plan)/60)
print("Standard deviation for planning: ", np.std(time_plan)/60)
