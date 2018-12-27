import networkx as nx
import os
from Partition_Scorer import partition_scorer
import numpy as np
import time
import progressbar
import pickle
from sklearn.cluster import KMeans
from pulp import *

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
#    constraint_sets = [set(group) for group in constraints]
#    constraint_maps = dict()    # Map node to a list of indices of constraints in CONSTRAINT_SETS of which that node is a part of
#    for node in graph.nodes():
#        constraint_indices = []
#        for i in range(len(constraint_sets)):
#            if node in constraint_sets[i]:
#                constraint_indices.append(i)
#        constraint_maps[node] = constraint_indices
#
    nodes = list(graph.nodes)
    """
    np.random.shuffle(nodes) # For randomization of order in which nodes are picked
    node_positions = dict() # Holds 2D position vectors for each node
    first_node_position = [0, 0]
    node_positions[nodes[0]] = first_node_position
    SCALE = len(nodes) * 4
    
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

        IGNORE_ROWDY_PROB = 0.25
        if rowdy_counter != 0 and np.random.rand() > IGNORE_ROWDY_PROB:
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
    """

    node_indices = dict()
    for i in range(len(nodes)):
        node_indices[nodes[i]] = i

    prob = LpProblem("prob", LpMinimize)

    ## Init LP variables ##
    lpVars = []     # Nxk array of LP variables where lpVars[i][j] is 1 if node i is in bus j
    for node in nodes:
        bus_subset_vars = []
        for j in range(num_buses):
            bus_subset_vars.append(LpVariable(node + "." + str(j), 0, 1))#, cat='Integer'))
        lpVars.append(bus_subset_vars)

    # Constraint slack variables
    slackVars = []
    for m in range(len(constraints)):
        slackVars.append(LpVariable("slack." + str(m), lowBound=0))

    ## Auxiliary vars: Because we want to have min |V_uj - V_wj| for (u, v) in E, we
    ## set V_uj - V_wj = Z+_uw,j - Z-_uw,j for positive Z+ and Z-, and find min Z+_uw,j + Z-_uw,j
    ## So for every pair of friends (u, w) we need two variables Z+_uw,j and Z-_uw,j for each bus
    ## j in 1,...,k
    auxVars = []    # |E|xk array of LP variable pairs ((Z+, Z-) for each friend pair for each bus)
    edges = graph.edges()
    for edge in edges:
        bus_subset_vars = []
        for j in range(num_buses):
            z_pos = LpVariable("Zpos" + edge[0] + edge[1] + "," + str(j), lowBound=0)
            z_neg = LpVariable("Zneg" + edge[0] + edge[1] + "," + str(j), lowBound=0)
            bus_subset_vars.append((z_pos, z_neg))
            u_index = node_indices[edge[0]]
            w_index = node_indices[edge[1]]
            u = lpVars[u_index][j]
            w = lpVars[w_index][j]
            
            constraint_lhs = LpAffineExpression([(u, 1), (w, -1), (z_pos, -1), (z_neg, 1)])
            prob += LpConstraint(constraint_lhs, LpConstraintEQ, rhs=0)
        auxVars.append(bus_subset_vars)

    ## Difference vars: because we want to have min -|V_u1 -...- V_uk| in order to encourage
    ## having only one bus be assigned a 1, we set V_u1 -...- V_uk = Z+_u - Z-_u for positive Z+
    ## and Z-, and find min -(Z+_u + Z-_u) = min -Z+_u - Z-_u
    difVars = [] # Nx1 array of (Z+, Z-) tuples, one for each node
    for i in range(len(nodes)):
        row = lpVars[i]
        z_pos = LpVariable("Zpos" + nodes[i], lowBound=0)
        z_neg = LpVariable("Zneg" + nodes[i], lowBound=0)
        difVars.append((z_pos, z_neg))
        constraint_lhs = [(z_pos, -1), (z_neg, 1), (row[0], 1)]
        for j in range(1, num_buses):
            constraint_lhs.append((row[j], -1))
        constraint_lhs = LpAffineExpression(constraint_lhs)
        prob += LpConstraint(constraint_lhs, LpConstraintEQ, rhs=0)



    ## Constraints ##
    # No bus should have <= than size_bus students in it and >= 1 student
    for j in range(num_buses):
        constraint_lhs = LpAffineExpression([(var_list[j], 1) for var_list in lpVars])
        prob += LpConstraint(constraint_lhs, LpConstraintLE, rhs=size_bus)
        prob += LpConstraint(constraint_lhs, LpConstraintGE, rhs=1)

    # The number of students in a rowdy group that are on the same bus should be less than the
    # size of the rowdy group, up to some amount of slack (i.e., rowdy group should not be subset
    # of any bus): SUM(V_ij) <= S + slackVar_m, for all j, over all m rowdy groups
    for m in range(len(constraints)):
        rowdy_group = constraints[m]
        member_indices = [node_indices[node] for node in rowdy_group]
        for j in range(num_buses):
            constraint_lhs = LpAffineExpression([(lpVars[i][j], 1) for i in member_indices] + [(slackVars[m], -1)])
            prob += LpConstraint(constraint_lhs, LpConstraintLE, rhs=(len(rowdy_group) - 1))

    # Should not have more than N students with a bus assigned
    flattened_vars = []
    for row in lpVars:
        flattened_vars += row
    constraint_lhs = [(var, 1) for var in flattened_vars]
    prob += LpConstraint(constraint_lhs, LpConstraintEQ, rhs=len(nodes))

    # Should have each student assigned to only one bus
    for row in lpVars:  # Row represents a student
        constraint_lhs = [(var, 1) for var in row]
        prob += LpConstraint(constraint_lhs, LpConstraintEQ, rhs=1)

    ## Objective ##
    objective = []
    for e in auxVars:
        for pair in e:
            objective += [(pair[0], 1), (pair[1], 1)]
    for s in slackVars:
        objective.append((s, 2))
    for pair in difVars:
        objective += [(pair[0], -1), (pair[1], -1)]

    prob += LpAffineExpression(objective)

    try:
        prob.solve()
#        print("HERE1")
#        prob.solve(pulp.PULP_CBC_CMD(maxSeconds=2, msg=0, fracGap=0))
#        print("HERE2")
        partitions = [[] for i in range(num_buses)]
        for i in range(len(nodes)):
            row = lpVars[i]
            for j in range(num_buses):
                var = row[j]
                if var.varValue >= 0.5:
                    partitions[j].append(nodes[i])
        return partitions
    except:
        print("Failed")
        return -1


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
    size_categories = ["small", "medium", "large"]
    if not os.path.isdir(path_to_outputs):
        os.mkdir(path_to_outputs)

    
    with open('bestPartitions.pkl', 'rb') as handle:
        bestPartitions = pickle.load(handle) # Stores size/input_name (e.g., small/135) as key, and [best partition score, best partition] as value

    for size in size_categories[:0]:
        print("Working on {} inputs...".format(size))
        category_path = path_to_inputs + "/" + size
        output_category_path = path_to_outputs + "/" + size
        category_dir = os.fsencode(category_path)

        if not os.path.isdir(output_category_path):
            os.mkdir(output_category_path)

        start = time.time()
        bar = progressbar.ProgressBar()
        for input_folder in bar(os.listdir(category_dir)[:20]):
            input_name = os.fsdecode(input_folder)
            graph, num_buses, size_bus, constraints = parse_input(category_path + "/" + input_name)
            solve(graph, num_buses, size_bus, constraints)

            solution = solve(graph, num_buses, size_bus, constraints)
            if solution == -1:
                continue
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
                        
        values = list(bestPartitions.values())
        scores = [v[0] for v in values]
        print("Total Score: %.4f" % np.mean(scores))



if __name__ == '__main__':
    main()


## Added for easy testing
category_path = path_to_inputs + "/" + "small"
category_dir = os.fsencode(category_path)

input_folder = os.listdir(category_dir)[0]
input_name = os.fsdecode(input_folder)
graph, num_buses, size_bus, constraints = parse_input(category_path + "/" + input_name)




