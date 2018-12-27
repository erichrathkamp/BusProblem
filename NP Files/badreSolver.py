import networkx as nx
import os
from Partition_Scorer import partition_scorer
import numpy as np
import time
import progressbar
import pickle
import random
from itertools import permutations
from sklearn.cluster import KMeans
from simanneal import Annealer

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
    graph.remove_edges_from(graph.selfloop_edges())   ## To remove self-edges
    parameters = open(folder_name + "/parameters.txt")
    num_buses = int(parameters.readline())
    size_bus = int(parameters.readline())
    constraints = []
    
    for line in parameters:
        line = line[1: -2]
        curr_constraint = [num.replace("'", "") for num in line.split(", ")]
        constraints.append(curr_constraint)

    return graph, num_buses, size_bus, constraints

def possible_partitions(nodes, num_buses, size_bus):
    if num_buses == 1:
        return [[nodes]]
    partitions = []
    for size in range(1, size_bus + 1):
        #check that this bus size if feasible
        #print("size: " + str(size))
        if(len(nodes) - size <= (num_buses - 1) * size_bus and size < len(nodes)):
            #print("passed check")
            #print("last part" + str(nodes[size:]))
            #print("first part" + str(nodes[:size]))
            for partition in possible_partitions(nodes[size:], num_buses - 1, size_bus):
                partition.append(nodes[:size])
                partitions.append(partition)
    return partitions

def brute_force(graph, num_buses, size_bus, constraints):
    
    #validConstraints = [c for c in constraints if len(c) <= size_bus]
    #constraints = validConstraints
    #shuffled list of permutations of the nodes
    all_possible_partitions = []
    best_partition = []
    max_score = -1
    nodes = list(graph.nodes())
    #print(nodes)
    #perms = permutations(nodes)
    #go through permutations creating possible outputs
    for _ in range(5):
        np.random.permutation(nodes)
        print("started new iteration")
        #nodes = next(perms)
        all_possible_partitions.extend(possible_partitions(nodes, num_buses, size_bus))
    print("started scoring")
    graph_copy = graph.copy()
    for i in range(min(2000, len(all_possible_partitions))):
    #for partition in all_possible_partitions:
        graph_copy = graph.copy()
        #check whether partition is better than the current best
        partition = all_possible_partitions[i]
        new_score = partition_scorer(partition, graph_copy, num_buses, size_bus, constraints)[0]
        if  new_score > max_score:
            best_partition = partition
            max_score = new_score
        print("score " + str(i) + " calculated")
    #print(max_score)
    best_partition = [[str(node) for node in bus] for bus in best_partition]
    return best_partition

def generate_random_buses(nodes, num_buses, size_bus, seed):
    buses = []
    random.seed(seed)
    #put one on each bus
    for _ in range(num_buses):
        if len(nodes) == 1:
            buses.append([nodes.pop()])
        else:
            index = random.randint(0, len(nodes) - 1)
            buses.append([nodes[index]])
            nodes.pop(index)

    #add the rest to each bus randomly so long as they don't violate size requirements
    for node in nodes:
        added = False
        while not added:
            bus = random.randint(0, num_buses - 1)
            if len(buses[bus]) < size_bus:
                buses[bus].append(node)
                added = True
    return buses

def brute_force_random(graph, num_buses, size_bus, constraints):
    #print("Started on an input")
    seed = random.SystemRandom()
    nodes = list(graph.nodes())
    all_possible_partitions = []
    for i in range(1000):
        all_possible_partitions.append(generate_random_buses(nodes.copy(), num_buses, size_bus, seed.random()))
    max_score = -1
    best_partition = 0
    for partition in all_possible_partitions:
        graph_copy = graph.copy()
        #check whether partition is better than the current best
        new_score = partition_scorer(partition, graph_copy, num_buses, size_bus, constraints)[0]
        if  new_score > max_score:
            best_partition = partition
            max_score = new_score
    #print("Score: " + str(max_score))
    #for i in range(num_buses):
        #print("Bus " + str(i) + ": " + str(best_partition[i]))
    return [[str(node) for node in bus] for bus in best_partition]
    

def solve(graph, num_buses, size_bus, constraints):
    #TODO: Write this method as you like. We'd recommend changing the arguments here as well

    validConstraints = [c for c in constraints if len(c) <= size_bus]
    constraints = validConstraints
    
    #######################
    ### Greedy solution ###
    #######################
    """
    remaining_nodes = set(graph.nodes)
    shortest_paths = dict(nx.all_pairs_bellman_ford_path_length(graph))
    starting_nodes = []
    first_node = np.random.choice(graph.nodes())
    starting_nodes.append(first_node)
    remaining_nodes.remove(first_node)
    while len(starting_nodes) < num_buses:
        max_distance = 0
        for node in remaining_nodes:
            total_dist = 0
            for s in starting_nodes:
                if (node in shortest_paths[s]):
                    total_dist += shortest_paths[s][node]
                else:
                    total_dist += 10000
            avg_dist = total_dist / len(starting_nodes)
            if (avg_dist > max_distance):
                max_distance = avg_dist
                furthest_node = node

        starting_nodes.append(furthest_node)
        remaining_nodes.remove(furthest_node)

    assignments = []
    nonFullBuses = []
    buses_finished = 0
    for s in starting_nodes:
        assignment = [s]
        starting_length = len(assignment)
        exist_new_neighbors = False
        while len(assignment) < size_bus and len(remaining_nodes) != 0:
            common_neighbors = remaining_nodes
            for node in assignment:
                new_common_neighbors = common_neighbors & set(graph.neighbors(node))
                if len(new_common_neighbors) != 0:
                    common_neighbors = new_common_neighbors
                    exist_new_neighbors = True

            if not exist_new_neighbors:
                    break
            new_node = np.random.choice(list(common_neighbors))
            assignment.append(new_node)
            remaining_nodes.remove(new_node)
    
        assignments.append(assignment)
        if len(assignment) < size_bus:
            nonFullBuses.append(buses_finished)
        buses_finished += 1

    while len(remaining_nodes) != 0:
        nextNode = np.random.choice(list(remaining_nodes))
        bus_id = np.random.choice(nonFullBuses)
        assignments[bus_id].append(nextNode)
        remaining_nodes.remove(nextNode)
        if len(assignments[bus_id]) == size_bus:
            nonFullBuses.remove(bus_id)

    return assignments
    """


    ########################
    ### K-Means Solution ###
    ########################
    constraint_sets = [set(group) for group in constraints]
    constraint_maps = dict()    # Map node to a list of indices of constraints in CONSTRAINT_SETS of which that node is a part of
    for node in graph.nodes():
        constraint_indices = []
        for i in range(len(constraint_sets)):
            if node in constraint_sets[i]:
                constraint_indices.append(i)
        constraint_maps[node] = constraint_indices
    
    nodes = list(graph.nodes)
    np.random.shuffle(nodes) # For randomization of order in which nodes are picked
    node_positions = dict() # Holds 2D position vectors for each node
    first_node_position = [0, 0]
    node_positions[nodes[0]] = first_node_position
    SCALE = 100

    for node in nodes[1:]:
        neighbors = graph.neighbors(node)
        x_sum, y_sum = 0, 0
        neighbor_counter = 0.0
        for neighbor in neighbors:
            if neighbor in node_positions:
                node_position = node_positions[neighbor]
                x_sum += node_position[0]
                y_sum += node_position[1]
                neighbor_counter += 1

        if neighbor_counter != 0:
            neighbor_centroid = [x_sum / neighbor_counter, y_sum / neighbor_counter]
        else:
            neighbor_centroid = [(np.random.rand() - 0.5) * 2*SCALE, (np.random.rand() - 0.5) * 2*SCALE]


        constraint_indices = constraint_maps[node]
        x_sum, y_sum = 0, 0
        rowdy_counter = 0.0
        for constraint_index in constraint_indices:
            constraint = constraint_sets[constraint_index]
            for n in constraint:
                if n in node_positions:
                    n_position = node_positions[n]
                    x_sum += n_position[0]
                    y_sum += n_position[1]
                    rowdy_counter += 1

        if rowdy_counter != 0:
            rowdy_centroid = [x_sum / rowdy_counter, y_sum / rowdy_counter]
            x_dist = rowdy_centroid[0] - neighbor_centroid[0]
            y_dist = rowdy_centroid[1] - neighbor_centroid[1]
            dist = np.sqrt((x_dist)**2 + (y_dist)**2)
            if dist != 0:
                repulsion_factor = (rowdy_counter * neighbor_counter) / (dist**2)
                node_pos_x = neighbor_centroid[0] - (repulsion_factor / dist) * (x_dist)
                node_pos_y = neighbor_centroid[1] - (repulsion_factor / dist) * (y_dist)
            else:
                if neighbor_counter == 0:
                    neighbor_counter = 1
                    print("********NEIGHBOR COUNTER 0")
                node_pos_x = neighbor_centroid[0] + (np.random.rand()-0.5) * SCALE * rowdy_counter / neighbor_counter
                node_pos_y = neighbor_centroid[1] + (np.random.rand()-0.5) * SCALE * rowdy_counter / neighbor_counter

            node_position = [node_pos_x, node_pos_y]

        else:
            node_position = neighbor_centroid

        node_positions[node] = node_position
            
    positions_list = [node_positions[node] for node in nodes]
    kmeansSolver = KMeans(n_clusters=num_buses, n_init=5)
    clusters = kmeansSolver.fit(positions_list)
    
    partitions = [[] for i in range(num_buses)]
    for i in range(len(nodes)):
        partition_id = clusters.labels_[i]
        partitions[partition_id].append(nodes[i])
    
    lengths = [len(p) for p in partitions]

    for i in range(len(lengths)):
        while lengths[i] > size_bus:
            partition = partitions[i]
            min_degree_node = findMinDegreeNode(partition, graph)

            smallest_bus = np.argmin(lengths)
            partitions[smallest_bus].append(min_degree_node)
            partitions[i].remove(min_degree_node)
            lengths[smallest_bus] += 1
            lengths[i] -= 1

        if lengths[i] == 0:
            max_length_id = np.argmax(lengths)
            partition = partitions[max_length_id]
            min_degree_node = findMinDegreeNode(partition, graph)
            partitions[i].append(min_degree_node)
            partitions[max_length_id].remove(min_degree_node)
            lengths[i] += 1
            lengths[max_length_id] -= 1
            
    return partitions

class Problem(Annealer):

    # pass extra data (the distance matrix) into the constructor
    def __init__(self, state, graph, num_buses, size_bus, constraints):
        self.graph = graph
        self.num_buses = num_buses
        self.size_bus = size_bus
        self.constraints = constraints
        super(Problem, self).__init__(state)  # important!

    def move(self):
        """Swaps vertices."""
        a = random.randint(0, self.num_buses - 1)
        b = random.randint(0, self.num_buses - 1)
        a_v = random.randint(0, len(self.state[a]) - 1)
        b_v = random.randint(0, len(self.state[b]) - 1)
        '''print("Number of buses: " + str(len(self.state)))
        print("length of target bus A: " + str(len(self.state[a])))
        print("length of target bus B: " + str(len(self.state[b])))
        print("index of vertex in A: " + str(a_v))
        print("index of vertex in B: " + str(b_v))'''
        self.state[a][a_v], self.state[b][b_v] = self.state[b][b_v], self.state[a][a_v]
    
    #we define how energy is computed (also known as the objective function):
    def energy(self):
        """Calculates the score of the partitions."""
        score = partition_scorer(self.state.copy(), self.graph.copy(), self.num_buses, self.size_bus, self.constraints)[0]
        return score

def sim_anneal(graph, num_buses, size_bus, constraints):
    problem = Problem(generate_random_buses(list(graph.nodes()), num_buses, size_bus, 100), graph, num_buses, size_bus, constraints)
    best_buses, best_energy = problem.anneal()
    return best_buses

def findMinDegreeNode(nodes_list, graph):
    min_degree = 10000
    for node in nodes_list:
        degree = graph.degree(node)
        if degree < min_degree:
            min_degree = degree
            min_degree_node = node

    return min_degree_node


def main():
    '''
    Main method which iterates over all inputs and calls `solve` on each.
    The student should modify `solve` to return their solution and modify
    the portion which writes it to a file to make sure their output is
    formatted correctly.
    '''
    size_categories = ["medium"]
    #size_categories = ["small", "medium", "large"]
    if not os.path.isdir(path_to_outputs):
        os.mkdir(path_to_outputs)

    
    with open('bestPartitions.pkl', 'rb') as handle:
        bestPartitions = pickle.load(handle) # Stores size/input_name (e.g., small/135) as key, and [best partition score, best partition] as value

    for size in size_categories:
        print("Working on {} inputs...".format(size))
        category_path = path_to_inputs + "/" + size
        output_category_path = path_to_outputs + "/" + size
        category_dir = os.fsencode(category_path)

        if not os.path.isdir(output_category_path):
            os.mkdir(output_category_path)

        start = time.time()
        bar = progressbar.ProgressBar()
        for input_folder in bar(os.listdir(category_dir)):
            input_name = os.fsdecode(input_folder)
            graph, num_buses, size_bus, constraints = parse_input(category_path + "/" + input_name)
#            solve(graph, num_buses, size_bus, constraints)
            '''if size == "small":
                solution = brute_force(graph, num_buses, size_bus, constraints)
            else:
                solution = solve(graph, num_buses, size_bus, constraints)'''
            solution = sim_anneal(graph, num_buses, size_bus, constraints)
            score, message = partition_scorer(solution, graph, num_buses, size_bus, constraints)
            
            keyName = size + "/" + input_name
            if score == -1:
                print("*******{}: {} - {}".format(keyName, score, message))

            if (keyName) not in bestPartitions:
                bestPartitions[keyName] = [score, solution]
            elif score > bestPartitions[keyName][0]:
                print("Found better score for {}/{}: {} vs. {}".format(size, input_name, score, bestPartitions[keyName][0]))
                bestPartitions[keyName] = [score, solution]

        end = time.time()
#            output_file = open(output_category_path + "/" + input_name + ".out", "w")
#
#            #TODO: modify this to write your solution to your
#            #      file properly as it might not be correct to
#            #      just write the variable solution to a file
#            output_file.write(solution)

#            output_file.close()

        print("Updating pickle file")
        with open('bestPartitions.pkl', 'wb') as handle:
            pickle.dump(bestPartitions, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    main()


## Added for easy testing
category_path = path_to_inputs + "/" + "small"
category_dir = os.fsencode(category_path)

input_folder = os.listdir(category_dir)[0]
input_name = os.fsdecode(input_folder)
graph, num_buses, size_bus, constraints = parse_input(category_path + "/" + input_name)




