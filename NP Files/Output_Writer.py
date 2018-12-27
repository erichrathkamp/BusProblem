import os
import pickle
import progressbar

path_to_small_outputs = "./small"
path_to_medium_outputs = "./medium"
path_to_large_outputs = "./large"

for path in [path_to_small_outputs, path_to_medium_outputs, path_to_large_outputs]:
    if not os.path.isdir(path):
        os.mkdir(path)

with open('bestPartitions.pkl', 'rb') as handle:
    bestPartitions = pickle.load(handle) # Stores size/input_name (e.g., small/135) as key, and [best partition score, best partition] as value

bar = progressbar.ProgressBar()
for key in bar(list(bestPartitions.keys())):
    assignments = bestPartitions[key][1]
    output_address = "./" + key + ".out"
    output_file = open(output_address, "w")
    for bus_assignment in assignments:
        output_file.write(str(bus_assignment) + "\n")

    output_file.close()

