import os
import pickle
import sys

class InputError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

def main():
    arguments = sys.argv[1:]
    if(len(arguments) <= 1):
        raise Exception("should pass at least 2 outputs") 
    print("Starting combiner")
    output_paths = []
    for arg in arguments:
        output_paths.append(arg)
    outputs = []
    for output_path in output_paths:
        #print(output_path)
        with open(str(output_path), 'rb') as handle:
            handle.seek(0)
            output = pickle.load(handle)
        outputs.append(output)
    outputs_length = len(list(outputs[0].keys())) 
    if not all([len(list(output.keys())) == outputs_length for output in outputs]):
        raise ValueError("Some output files are incomplete")
    combined = {}
    print("Comparing solutions from all files")
    for key in outputs[0].keys():
        solutions = [output[key] for output in outputs]
        best_solution = max(solutions, key = lambda x: x[0])
        combined[key] = best_solution
    print("Writing best outputs to combinedPartitions.pkl")
    with open('combinedPartitions.pkl', 'wb') as handle:
        pickle.dump(combined, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()