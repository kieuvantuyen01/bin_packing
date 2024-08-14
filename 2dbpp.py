from itertools import chain, combinations
import math
from threading import Timer

from pysat.formula import CNF
from pysat.solvers import Glucose3

import matplotlib.pyplot as plt
import timeit

class TimeoutException(Exception): pass
# Initialize the CNF formula

#read file
def read_file_instance(filepath):
    f = open(filepath)
    return f.read().splitlines()

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

input = read_file_instance("inputs/test.txt")
n = int(input[0])
bin_size = input[1].split()
W = int(bin_size[0])
H = int(bin_size[1])
items = [[int(val) for val in i.split()] for i in input[2:]]
items_area = [i[0] * i[1] for i in items]
bin_area = W * H
lower_bound = math.ceil(sum(items_area) / bin_area)
max_bin = min(n, 2 * lower_bound)
upper_bound = min(n, 2 * lower_bound)
counter = 1

def positive_range(end):
    if (end < 0):
        return []
    return range(end)

def OPP(rectangles, width, height):
# Define the variables
    cnf = CNF()
    variables = {}
    opp_counter = 1

    # find max height and width rectangles for largest rectangle symmetry breaking
    max_height = max([int(rectangle[1]) for rectangle in rectangles])
    max_width = max([int(rectangle[0]) for rectangle in rectangles])

    # create lr, ud variables
    for i in range(len(rectangles)):
        for j in range(len(rectangles)):
            variables[f"lr{i + 1},{j + 1}"] = opp_counter  # lri,rj
            opp_counter += 1
            variables[f"ud{i + 1},{j + 1}"] = opp_counter  # uri,rj
            opp_counter += 1
        for e in positive_range(width - rectangles[i][0] + 2):
            variables[f"px{i + 1},{e}"] = opp_counter  # pxi,e
            opp_counter += 1
        for f in positive_range(height - rectangles[i][1] + 2):
            variables[f"py{i + 1},{f}"] = opp_counter  # pyi,f
            opp_counter += 1

    # Add the 2-literal axiom clauses (order constraint)
    for i in range(len(rectangles)):
       # ¬pxi,e ∨ pxi,e+1
        for e in range(width - rectangles[i][0] + 1):  # -1 because we're using e+1 in the clause
            cnf.append([-variables[f"px{i + 1},{e}"],
                        variables[f"px{i + 1},{e + 1}"]])
        #  ¬pyi,f ∨ pxi,f+1
        for f in range(height - rectangles[i][1] + 1):  # -1 because we're using f+1 in the clause
            cnf.append([-variables[f"py{i + 1},{f}"],
                        variables[f"py{i + 1},{f + 1}"]])


    # Add the 3-literal non-overlapping constraints
    def non_overlapping(i, j, h1, h2, v1, v2):
        i_width = rectangles[i][0]
        i_height = rectangles[i][1]
        j_width = rectangles[j][0]
        j_height = rectangles[j][1]

        # lri, j ∨ lrj, i ∨ udi, j ∨ udj, i
        four_literal = []
        if h1: four_literal.append(variables[f"lr{i + 1},{j + 1}"])
        if h2: four_literal.append(variables[f"lr{j + 1},{i + 1}"])
        if v1: four_literal.append(variables[f"ud{i + 1},{j + 1}"])
        if v2: four_literal.append(variables[f"ud{j + 1},{i + 1}"])
        cnf.append(four_literal)

        # ¬lri, j ∨ ¬pxj, e
        if h1:
            for e in range(i_width):
                if f"px{j + 1},{e}" in variables:
                    cnf.append([-variables[f"lr{i + 1},{j + 1}"],
                                -variables[f"px{j + 1},{e}"]])
        # ¬lrj,i ∨ ¬pxi,e
        if h2:
            for e in range(j_width):
                if f"px{i + 1},{e}" in variables:
                    cnf.append([-variables[f"lr{j + 1},{i + 1}"],
                                -variables[f"px{i + 1},{e}"]])
        # ¬udi,j ∨ ¬pyj,f
        if v1:
            for f in range(i_height):
                if f"py{j + 1},{f}" in variables:
                    cnf.append([-variables[f"ud{i + 1},{j + 1}"],
                                -variables[f"py{j + 1},{f}"]])
        # ¬udj, i ∨ ¬pyi, f,
        if v2:
            for f in range(j_height):
                if f"py{i + 1},{f}" in variables:
                    cnf.append([-variables[f"ud{j + 1},{i + 1}"],
                                -variables[f"py{i + 1},{f}"]])

        for e in positive_range(width - i_width):
            # ¬lri,j ∨ ¬pxj,e+wi ∨ pxi,e
            if h1:
                if f"px{j + 1},{e + i_width}" in variables:
                    cnf.append([-variables[f"lr{i + 1},{j + 1}"],
                                variables[f"px{i + 1},{e}"],
                                -variables[f"px{j + 1},{e + i_width}"]])
            # ¬lrj,i ∨ ¬pxi,e+wj ∨ pxj,e
            if h2:
                if f"px{i + 1},{e + j_width}" in variables:
                    cnf.append([-variables[f"lr{j + 1},{i + 1}"],
                                variables[f"px{j + 1},{e}"],
                                -variables[f"px{i + 1},{e + j_width}"]])

        for f in positive_range(height - i_height):
            # udi,j ∨ ¬pyj,f+hi ∨ pxi,e
            if v1:
                if f"py{j + 1},{f + i_height}" in variables:
                    cnf.append([-variables[f"ud{i + 1},{j + 1}"],
                                variables[f"py{i + 1},{f}"],
                                -variables[f"py{j + 1},{f + i_height}"]])
            # ¬udj,i ∨ ¬pyi,f+hj ∨ pxj,f
            if v2:
                if f"py{i + 1},{f + j_height}" in variables:
                    cnf.append([-variables[f"ud{j + 1},{i + 1}"],
                                variables[f"py{j + 1},{f}"],
                                -variables[f"py{i + 1},{f + j_height}"]])

    for i in range(len(rectangles)):
        for j in range(i + 1, len(rectangles)):
            # lri,j ∨ lrj,i ∨ udi,j ∨ udj,i
            #Large-rectangles horizontal
            if rectangles[i][0] + rectangles[j][0] > width:
                non_overlapping(i, j, False, False, True, True)

            #Large-rectangles vertical
            if rectangles[i][1] + rectangles[j][1] > height:
                non_overlapping(i, j, True, True, False, False)

            #Same-sized rectangles
            elif rectangles[i] == rectangles[j]:
                non_overlapping(i, j, True, False, True, True)
            #
            #largest width rectangle
            elif rectangles[i][0] == max_width and rectangles[j][0] > (width - max_width) / 2:
                non_overlapping(i, j, False, True, True, True)
            #
            #largest height rectangle
            elif rectangles[i][1] == max_height and rectangles[j][1] > (height - max_height) / 2:
                non_overlapping(i, j, True, True, False, True)

           #normal rectangles
            else:
                non_overlapping(i, j, True, True, True, True)

    # Domain encoding for px and py: 0 <= x <= width and 0 <= y <= height
    # equal to: px(i, W-wi) ^ !px(i,-1) and py(i, H-hi) ^ !py(i,-1)
    for i in range(len(rectangles)):
        cnf.append([variables[f"px{i + 1},{width - rectangles[i][0]}"]]) # px(i, W-wi)
        cnf.append([variables[f"py{i + 1},{height - rectangles[i][1]}"]])  # py(i, H-hi)
    result = None
    with Glucose3() as solver:
        solver.append_formula(cnf)
        if solver.solve():
            result = "Sat"
        else:
            result = "Unsat"
    return result

def bpp():
    cnf = CNF()
    variables = {}
    global counter
    global upper_bound
    # x_i_j
    for i in range(n):
        for j in range(upper_bound):
            variables[f"x{i + 1},{j + 1}"] = counter
            counter += 1
    # k_j
    for i in range(upper_bound):
        variables[f"k{i + 1}"] = counter
        counter += 1
        
    #  Each item must be in one bin and one bin only
    for i in range(n):
        cnf.append([variables[f"x{i + 1},{j + 1}"] for j in range(upper_bound)])  # At least one bin
        for j1 in range(upper_bound):
            for j2 in range(j1 + 1, upper_bound):
                cnf.append([-variables[f"x{i + 1},{j1 + 1}"], -variables[f"x{i + 1},{j2 + 1}"]])
    
    # If x_i_j is true, then k_j must be true
    for j in range(upper_bound):
        for i in range(n):
            cnf.append([-variables[f"x{i + 1},{j + 1}"], variables[f"k{j + 1}"]])
    
    for j in range(upper_bound):
        for subset in powerset(range(n)):
            if subset and len(subset) != 1:
                opp = OPP([items[i] for i in subset], W, H)
                # If the subset is infeasible, then the subset cannot be in the same bin
                if opp == "Unsat":
                    cnf.append([-variables[f"x{i + 1},{j + 1}"] for i in subset])
                print(subset, j + 1, [variables[f"x{i + 1},{j + 1}"] for i in subset])
                
    print("Variables:", variables)
    print(cnf.clauses)
    timeout = 60  # timeout in seconds
    def interrupt(solver):
        solver.interrupt()


    with Glucose3(bootstrap_with=cnf.clauses, use_timer=True) as s:
        timer = Timer(timeout, interrupt, [s])
        timer.start()
        try:
            for _ in range(upper_bound):
                result = s.solve(assumptions=[-variables[f"k{j + 1}"] for j in range(upper_bound)])
                if result:
                    print("SAT")
                    timer.cancel()
                    break
                else:
                    upper_bound -= 1
            else:
                print("UNSAT")
            timer.cancel()
        except TimeoutException:
            print("Timeout")
        # Print solution to console
        if result:
            print("Number of bins:", upper_bound)
            model = s.get_model()
            # print(model)
            if model is not None:
                print(model)
                print(variables)
                for k in range(max_bin):
                    if model[variables[f"k{k + 1}"] - 1] > 0:
                        print(f"Bin {k + 1}:")
                        for i in range(n):
                            if model[variables[f"x{i + 1},{k + 1}"] - 1] > 0:
                                print(f"Item {i + 1}:", items[i])
                
# 
bpp()


    
