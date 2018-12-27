import networkx as nx
import matplotlib.pyplot as plt

def partition_scorer(assignments, graph, num_buses, size_bus, constraints):

    if len(assignments) != num_buses:
        return -1, "Must assign students to exactly {} buses, found {} buses".format(num_buses, len(assignments))
    
    # make sure no bus is empty or above capacity
    for i in range(len(assignments)):
        if len(assignments[i]) > size_bus:
            return -1, "Bus {} is above capacity".format(i)
        if len(assignments[i]) <= 0:
            return -1, "Bus {} is empty".format(i)

    bus_assignments = {}
    attendance_count = 0
    
    # make sure each student is in exactly one bus
    attendance = {student:False for student in graph.nodes()}
    for i in range(len(assignments)):
        if not all([student in graph for student in assignments[i]]):
            return -1, "Bus {} references a non-existent student: {}".format(i, assignments[i])
    
        for student in assignments[i]:
            # if a student appears more than once
            if attendance[student] == True:
                print(assignments[i])
                return -1, "{0} appears more than once in the bus assignments".format(student)
            
            attendance[student] = True
            bus_assignments[student] = i

    # make sure each student is accounted for
    if not all(attendance.values()):
        return -1, "Not all students have been assigned a bus"
    
    total_edges = graph.number_of_edges()
    # Remove nodes for rowdy groups which were not broken up
    for i in range(len(constraints)):
        busses = set()
        for student in constraints[i]:
            busses.add(bus_assignments[student])
        if len(busses) <= 1:
            for student in constraints[i]:
                if student in graph:
                    graph.remove_node(student)

    # score output
    score = 0
    for edge in graph.edges():
        if bus_assignments[edge[0]] == bus_assignments[edge[1]]:
            score += 1
    score = score / total_edges
    
    
    return score, "Valid output submitted with score: {}".format(score)
