import networkx as nx
from functools import reduce
import os
import time

###########################################
# Change this variable to the path to 
# the folder containing all three input
# size category folders
###########################################
path_to_inputs = "./all_inputs"

###########################################
# Change this variable if you want
# your outputs to be put in a 
# different folder
###########################################
path_to_outputs = "./outputs"

def parse_input(folder_name):
    '''
        Parses an input and returns the corresponding graph and parameters

        Inputs:
            folder_name - a string representing the path to the input folder

        Outputs:
            (graph, num_buses, size_bus, constraints)
            graph - the graph as a NetworkX object
            num_buses - an integer representing the number of buses you can allocate to
            size_buses - an integer representing the number of students that can fit on a bus
            constraints - a list where each element is a list vertices which represents a single rowdy group
    '''
    graph = nx.read_gml(folder_name + "/graph.gml")
    parameters = open(folder_name + "/parameters.txt")
    num_buses = int(parameters.readline())
    size_bus = int(parameters.readline())
    constraints = []
    
    for line in parameters:
        line = line[1: -2]
        curr_constraint = [num.replace("'", "") for num in line.split(", ")]
        constraints.append(curr_constraint)

    return graph, num_buses, size_bus, constraints

def brute_force(graph, num_buses, size_bus, constraints):
    #TODO: can use bruteforce for small inputs
    pass

def create_buses(graph, num_buses, size_bus, constraints):
    #print("CREATING BUSES")
    buses = []
    G_prime = graph.copy()
    #iteratively run min-cut some amount of times where k = num_buses, add the smaller partition to buses
    while len(buses) < num_buses:
        #print("start of iteration")
        components = sorted(nx.connected_components(G_prime), key=len, reverse = True)
        while len(components) > 1:
            buses.append(list(components.pop()))
            if len(buses) == (num_buses - 1):
                #print("combining remaining components")
                buses.append(list(reduce(lambda x, y: x | y, components)))
                return buses
        G_prime = nx.Graph(G_prime.subgraph(components[0]))
        min_cut = nx.minimum_edge_cut(G_prime)
        G_prime.remove_edges_from(min_cut)
        components = sorted(nx.connected_components(G_prime), key=len)
        if len(components) == 1:
            #print("couldn't cut")
            buses.append(list(components[0]))
            G = nx.Graph(graph.subgraph(buses.pop(buses.index(max(buses, key = len)))))
            G_prime = nx.Graph(graph.subgraph(sorted(nx.connected_components(G), key = len, reverse = True)[0]))
            pass
        #print("Number of components: " +str(len(components)))
        #print("Number of buses: " +str(len(buses)))
        #print("Desired number of buses: " +str(num_buses))
        #print("Component 0 size: " + str(len(components[0])))
        buses.append(list(components[0]))
        G_prime = nx.Graph(G_prime.subgraph(components[1]))
    return buses
    
def solve(graph, num_buses, size_bus, constraints):
    '''
        Runs min cut on the graph until there are num_buses groups, redistributes vertices 
        in order to meet size constraints. Then iterates over each bus and 
        optimizes by swapping vertex with other bus in order to improve score/reduce rowdy groups. Stops
        after iterating over all buses. 
    '''
    buses = create_buses(graph, num_buses, size_bus, constraints)
    #print("ACTUAL NUMBER OF BUSES: " + str(len(buses)))
    #print(str(buses))
    #process the buses with extra vertices
    for bus in buses: 
        while len(bus) > size_bus:
            candidate = min(bus, key = lambda v: bus_degree(v, bus, graph))
            #other_buses = buses.copy()
            #other_buses.remove(bus)
            #find bus with maximum indegree and creates no rowdy and size of the other bus is also less than size_bus
            #insert if can
            best_insert = 0
            best_bus_degree = 0
            for i in range(num_buses):
                if bus_degree(candidate, buses[i], graph) >= best_bus_degree \
                    and len(buses[i]) < size_bus and buses[i] != bus:
                        best_insert = i
            buses[best_insert].append(candidate)
            bus.remove(candidate)                    
    
    #begin optimizing by swapping
    return buses

def bus_degree(vertex, bus, graph):
    #print(vertex)
    #print("Degree: " + str(graph.subgraph(bus).degree(vertex)))
    degree = graph.subgraph(bus + [vertex]).degree(vertex)
    if not degree: 
        return 0
    return degree

def check_no_rowdy(vertex, bus):
    '''
        checks that adding vertex to bus does not create a rowdy group
    ''' 
    swapped = bus.copy()
    swapped.append(vertex)
    relevant_rowdy = filter(lambda x: vertex in x, constraints)
    for rowdy in relevant_rowdy:
        if set(rowdy) < set(swapped):
            return False
    return True

def main():
    '''
        Main method which iterates over all inputs and calls `solve` on each.
        The student should modify `solve` to return their solution and modify
        the portion which writes it to a file to make sure their output is
        formatted correctly.
    '''
    start = time.time()
    size_categories = ["small", "medium", "large"]
    if not os.path.isdir(path_to_outputs):
        os.mkdir(path_to_outputs)

    for size in size_categories:
        category_path = path_to_inputs + "/" + size
        output_category_path = path_to_outputs + "/" + size
        category_dir = os.fsencode(category_path)
        
        if not os.path.isdir(output_category_path):
            os.mkdir(output_category_path)

        for input_folder in os.listdir(category_dir):
            input_name = os.fsdecode(input_folder) 
            graph, num_buses, size_bus, constraints = parse_input(category_path + "/" + input_name)
            solution = solve(graph, num_buses, size_bus, constraints)
            output_file = open(output_category_path + "/" + input_name + ".out", "w")

            #TODO: modify this to write your solution to your 
            #      file properly as it might not be correct to 
            #      just write the variable solution to a file
            output_file.write(str(solution))

            output_file.close()
        now = time.time()
        print("Done with " + size + " inputs!")
        print("Completed " + size + " inputs in " + str(now - start) + " seconds!")
        start = now
if __name__ == '__main__':
    main()


