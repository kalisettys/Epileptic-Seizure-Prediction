import pyedflib
import numpy as np
from scipy import fftpack
import os
import pandas as pd

import csv

#loading the dataset using PyEDFlib toolbox in python === https://pyedflib.readthedocs.io/en/latest/

#reads the data from edf format
def preprocess(filename = "project635/case01/chb01_01.edf"):

    num_of_rows = 0
    seizure_count = 0
    csv_output_file = filename.split(".")[0] + ".csv"
    num_of_runs = 150
    speed_up = []

    with open(csv_output_file,"w") as output_file, open("project635/case01/patient_input.csv") as input_file:
        #saves all the 432 features to a csv for all 18 channels
        write_to_csv = csv.writer(output_file, delimiter=",")

        #read input_file file with seizure data about each edf file
        read_csv = csv.reader(input_file, delimiter=",")
        for line in read_csv:
            if line[0] == filename.split("/")[-1]:
                seizure_start_time = int(line[1])
                seizure_end_time = int(line[2])
                seizures = int(line[3])

        f = pyedflib.EdfReader(filename)
        n = f.signals_in_file

        signal_labels = f.getSignalLabels()
        #print(signal_labels)

        # initializing variables
        data_points_per_second = 256
        num_of_windows = 3
        size_per_window = 2

        num_of_channels = np.zeros((n, f.getNSamples()[0]))[:18]

        #considering only the first 18 channels of the 23 available
        for i in np.arange(n)[:18]:
            num_of_channels[i, :] = f.readSignal(i)

        # looking at only the 6 seconds of the entire time period and going by 2 second epochs
        for start_point in range(0,len(num_of_channels[0])-(data_points_per_second*num_of_windows*size_per_window), data_points_per_second):
            list_of_features = []

            #appending 3, 2 sec windows in window holder
            for channel in num_of_channels:
                first_six_seconds = channel[start_point: start_point + (data_points_per_second * num_of_windows * size_per_window)]
                two_second_windows = np.array_split(first_six_seconds, 3)

                ##3 lists (windows) each with 24 values PER 2 second windows = 24, 24, 24
                windows_with_fft = []
                for window in two_second_windows:
                    perform_fft = [abs(value)
                                   for value in fftpack.rfft(window)[0:24]]
                    windows_with_fft.append(perform_fft)

                #input(len(windows_with_fft))

                for window_with_fft in windows_with_fft:
                    each_block_points = np.array_split(window_with_fft, 8)

                    for each_block_point in each_block_points:
                        extracted_feature = (sum(each_block_point)/(len(each_block_point)))**2

                        list_of_features.append(extracted_feature)
            ##classifying seizure vs non-seizure records
            if seizures == 1 and (
                    seizure_start_time <= start_point <= seizure_end_time
                 or
                    seizure_start_time <= (start_point+1536) <= seizure_end_time
            ):
                list_of_features.append(1)
                seizure_count+=1
            else:
                list_of_features.append(0)

            #add list of features to the kind of like
            speed_up.append(list_of_features)
            if len(speed_up) >= num_of_runs:
                write_to_csv.writerows(speed_up)
                speed_up = []
            num_of_rows += 1

        if len(speed_up) < 150:
            write_to_csv.writerows(speed_up)


    print("Total Lines: {0}".format(num_of_rows))
    print("Total Seizure Records: {0}".format(seizure_count))


all_files = os.listdir("project635/case01")

myFiles = []

for each_file in os.listdir("project635/case01"):

    if each_file.endswith(".edf"):
        x = os.path.join("project635/case01", each_file)
        myFiles.append(x)

for file in sorted(myFiles):
    print("\n\nStarting File: {0}".format(file))
    preprocess(file)






