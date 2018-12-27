import networkx as nx
import os
from Partition_Scorer import partition_scorer
import numpy as np
import time
import progressbar
import pickle
from sklearn.cluster import KMeans
#from pulp import *
import mosek, sys

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

inf = 0.0
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()

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
    ##############
    ## MOSEK LP ##
    ##############
    print("started LP")
    nodes = list(graph.nodes)
    edges = graph.edges()
    node_indices = dict()
    for i in range(len(nodes)):
        node_indices[nodes[i]] = i

    with mosek.Env() as env:
        with env.Task(0, 0) as task:
            # Attach a log stream printer to the task
            task.set_Stream(mosek.streamtype.log, streamprinter)
            # Bound keys for constraints
            bkc = []
            # Bound values for constraints
            blc = []
            buc = []
            # Bound keys for variables
            bkx = []
            # Bound values for variables
            blx = []
            bux = []
            #add variables
            #task.appendvars(len(nodes) * num_buses)
            #slack variables
            #task.appendvars(len(constraints))

            num_vertex = len(nodes) * num_buses
            num_slack = len(constraints)

            ## Auxiliary vars: Because we want to have min |V_uj - V_wj| for (u, v) in E, we
            ## set V_uj - V_wj = Z+_uw,j - Z-_uw,j for positive Z+ and Z-, and find min Z+_uw,j + Z-_uw,j
            ## So for every pair of friends (u, w) we need two variables Z+_uw,j and Z-_uw,j for each bus
            ## j in 1,...,k
            # |E|xk array of LP variable pairs ((Z+, Z-) for each friend pair for each bus)
            #task.appendvars(2 * len(edges) * num_buses)

            ## Difference vars: because we want to have min -|V_u1 -...- V_uk| in order to encourage
            ## having only one bus be assigned a 1, we set V_u1 -...- V_uk = Z+_u - Z-_u for positive Z+
            ## and Z-, and find min -(Z+_u + Z-_u) = min -Z+_u - Z-_u
            #task.appendvars(2 * len(nodes))

            

            #for node in range(len(nodes)):
                #for bus in range(0, size_bus):

            num_auxiliary = 2 * len(edges) * num_buses
            num_difference = 2 * len(nodes)

            #define bounds on variables for each vertex
            for _ in range(len(nodes) * num_buses):
                bkx.append(mosek.boundkey.ra)
                blx.append(0.0)
                bux.append(1.0)
            #define bounds on slack variables
            for _ in range(len(constraints)):
                bkx.append(mosek.boundkey.lo)
                blx.append(0.0)  
                bux.append(+inf)
            #define bounds on auxiliary vars 
            for _ in range(num_auxiliary):
                bkx.append(mosek.boundkey.lo)
                blx.append(0.0)  
                bux.append(+inf)
            '''
            #define bounds on difference vars 
            for _ in range(num_difference):
                bkx.append(mosek.boundkey.lo)
                blx.append(0.0)  
                bux.append(+inf)
            '''

            #constraint matrix
            asub = []
            aval = []
        
            #offsets for where variables begin
            offset_slack = num_vertex
            offset_aux = offset_slack + num_slack
            offset_diff = offset_aux + num_auxiliary

            #define constraints using auxiliary vars
            for _ in range(len(edges) * num_buses):
                bkc.append(mosek.boundkey.fx)
                blc.append(0.0)
                buc.append(0.0)

            for i, edge in enumerate(edges):
                for j in range(num_buses):
                    u_index = node_indices[edge[0]]
                    w_index = node_indices[edge[1]]
                    vertex_u = (j * len(nodes) + u_index)
                    vertex_w = (j * len(nodes) + w_index)
                    aux_e_pos = offset_aux + (j * len(edges) * 2) + (i * 2)
                    aux_e_neg = offset_aux + (j * len(edges) * 2) + (i * 2) + 1
                    # add constraint u - w - z_edge+ + z_edge- to A matrix
                    asub.append([vertex_u, vertex_w, aux_e_pos, aux_e_neg])
                    aval.append([1.0, -1.0, -1.0, 1])

            '''#define constraints using difference vars
            for _ in range(len(nodes)):
                bkc.append(mosek.boundkey.fx)
                blc.append(0.0)
                buc.append(0.0)

            for i in range(len(nodes)):
                #add constraint - z_node+ + z_node- + all  of the xj for node x where j is bus index 
                diff_pos = offset_diff + (i * 2)
                diff_neg = offset_diff + (i * 2) + 1 
                asub.append([diff_pos, diff_neg] + list(range(i, num_vertex, len(nodes))))
                aval.append([-1, 1, 1] + [-1 for _ in range(num_buses - 1)])
            '''
            #define constraints on bus capacity and minimum bus occupancy
            for _ in range(num_buses):
                bkc.append(mosek.boundkey.ra)
                blc.append(1.0)
                buc.append(size_bus)

            for j in range(num_buses):
                asub.append(list(range(j * len(nodes), (j + 1) * len(nodes))))
                aval.append([1 for _ in range(len(nodes))])

            #LP constraints for rowdy groups
            for m in range(len(constraints)):
                rowdy_group = constraints[m]
                member_indices = [node_indices[node] for node in rowdy_group]
                for j in range(num_buses):
                    asub.append([(j * len(nodes) + i) for i in member_indices] + [(offset_slack + m)])
                    aval.append([1 for _ in range(len(member_indices))] + [-1])
                    bkc.append(mosek.boundkey.up)
                    blc.append(-inf)
                    buc.append(len(rowdy_group) - 1)

            #constraint to make sure all N have bus assigned
            asub.append(list(range(num_vertex)))
            aval.append([1 for _ in range(num_vertex)])
            bkc.append(mosek.boundkey.fx)
            blc.append(len(nodes))
            buc.append(len(nodes))

            #constraints to make sure each student is assigned to one bus
            for i in range(len(nodes)):
                asub.append(list(range(i, num_vertex, len(nodes))))
                aval.append([1 for _ in range(num_buses)])
                bkc.append(mosek.boundkey.fx)
                blc.append(1)
                buc.append(1)

            #coefficients for objective function
            c = [0 for _ in range(num_vertex)] + [1 for _ in range(num_slack)] + [1 for _ in range(num_auxiliary)]

            #print("COEFFICIENTS LENGTH: " + str(len(c)))

            numvar = len(bkx)
            numcon = len(bkc)
            task.appendvars(numvar)
            task.appendcons(numcon)

            #constrain vertex variables to integer values
            task.putvartypelist(list(range(num_vertex)),
                                [mosek.variabletype.type_int
                                 for _ in range(num_vertex)])
            '''print("NUMBER OF VARIABLES: " + str(numvar))
            print("NUMBER OF CONSTRAINTS: " + str(numcon))
            print("DIMENSION OF ASUB: " + str(len(asub)))
            print("DIMENSION OF AVAL: " + str(len(aval)))
            print(any([len(x) != len(y) for x, y in zip(asub, aval)]))'''
            for j in range(numvar):
                # Set the linear term c_j in the objective.
                task.putcj(j, c[j])

                # Set the bounds on variable j
                # blx[j] <= x_j <= bux[j]
                task.putvarbound(j, bkx[j], blx[j], bux[j])
            for i in range(numcon):
                # Input row i of A
                #print("fine so far: " + str(i))
                task.putarow(i, asub[i], aval[i])  
                #put bounds on constraints
                task.putconbound(i, bkc[i], blc[i], buc[i])          

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.minimize)

            try:
                # Solve the problem
                task.optimize()
                # Print a summary containing information
                # about the solution for debugging purposes
                task.solutionsummary(mosek.streamtype.msg)

                # Get status information about the solution
                solsta = task.getsolsta(mosek.soltype.bas)
                
                if (solsta == mosek.solsta.optimal or
                        solsta == mosek.solsta.near_optimal):
                    xx = [0.] * numvar
                    task.getxx(mosek.soltype.bas, # Request the basic solution.
                                xx)
                    print("Optimal solution: ")
                    for i in range(numvar):
                        print("x[" + str(i) + "]=" + str(xx[i]))
                    partitions = [[] for i in range(num_buses)]
                    for i in range(len(nodes)):
                        for j in range(num_buses):
                            var = xx[j * len(nodes) + i]
                            if var >= 0.5:
                                partitions[j].append(nodes[i])
                    print("SUCCESS")
                    return partitions
                elif (solsta == mosek.solsta.dual_infeas_cer or
                        solsta == mosek.solsta.prim_infeas_cer or
                        solsta == mosek.solsta.near_dual_infeas_cer or
                        solsta == mosek.solsta.near_prim_infeas_cer):
                    print("Primal or dual infeasibility certificate found.\n")
                elif solsta == mosek.solsta.unknown:
                    print("Unknown solution status")
                else:
                    print("Other solution status") 
            except: 
                print("Failed")
                return -1





'''
    ##############
    ## PULP LP ##
    ##############
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
'''

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


    for size in size_categories:
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




