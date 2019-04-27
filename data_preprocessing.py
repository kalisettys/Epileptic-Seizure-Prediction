import pyedflib
import numpy as np
from scipy import fftpack
import os

import csv

#loading the dataset using PyEDFlib toolbox in python === https://pyedflib.readthedocs.io/en/latest/

#reads the data from edf format
def preprocess(filename="project635/case01/chb01_01.edf"):

    num_of_rows = 0
    csv_output_file = filename.split(".")[0] + ".csv"

    with open(csv_output_file, "w") as output_file:
        write_to_csv = csv.writer(output_file, delimiter=",")

    with open("project635/case01/patient_input.csv") as input_file:
        read_csv = csv.reader(input_file, delimiter=",")




        f = pyedflib.EdfReader(filename)
        n = f.signals_in_file

        signal_labels = f.getSignalLabels()
        print(signal_labels)

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
                #input(len(first_six_seconds))

                two_second_windows = np.array_split(first_six_seconds, 3)

                ##3 lists each with 24 values for 2 second windows
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

            write_to_csv.writerow(list_of_features)
            num_of_rows += 1

    print("Total Lines: {0}".format(num_of_rows))


####read csv files


preprocess()











# all_files = os.listdir("project635/case01")
#
# myFiles = []
#
# for each_file in os.listdir("project635/case01"):
#
#     if each_file.endswith(".edf"):
#         x = os.path.join("project635/case01", each_file)
#         myFiles.append(x)
#
# for file in sorted(myFiles):
#     print(file)
