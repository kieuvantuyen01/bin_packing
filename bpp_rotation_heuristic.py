from itertools import chain, combinations
import math
from threading import Timer

from pysat.formula import CNF
from pysat.solvers import Solver

import matplotlib.pyplot as plt
import timeit
from typing import List, Tuple

class Rectangle:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

class FreeRectangle(Rectangle):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(width, height)
        self.x = x
        self.y = y

class Bin:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.free_rectangles = [FreeRectangle(0, 0, width, height)]
        self.items = []

def guillotine_split(free_rect: FreeRectangle, item: Rectangle) -> List[FreeRectangle]:
    remaining_width = free_rect.width - item.width
    remaining_height = free_rect.height - item.height
    new_rectangles = []

    if remaining_width > 0:
        new_rectangles.append(FreeRectangle(
            free_rect.x + item.width,
            free_rect.y,
            remaining_width,
            free_rect.height
        ))
    
    if remaining_height > 0:
        new_rectangles.append(FreeRectangle(
            free_rect.x,
            free_rect.y + item.height,
            item.width,
            remaining_height
        ))

    return new_rectangles

def merge_rectangles(free_rectangles: List[FreeRectangle]) -> List[FreeRectangle]:
    merged = []
    for rect in free_rectangles:
        if not any(r for r in merged if r.x <= rect.x and r.y <= rect.y and 
                   r.x + r.width >= rect.x + rect.width and 
                   r.y + r.height >= rect.y + rect.height):
            merged.append(rect)
    return merged

def maximal_rectangles_split(free_rect: FreeRectangle, item: Rectangle) -> List[FreeRectangle]:
    new_rectangles = []
    
    # Right rectangle
    if free_rect.x + free_rect.width > item.x + item.width:
        new_rectangles.append(FreeRectangle(
            item.x + item.width,
            free_rect.y,
            free_rect.x + free_rect.width - (item.x + item.width),
            free_rect.height
        ))
    
    # Top rectangle
    if free_rect.y + free_rect.height > item.y + item.height:
        new_rectangles.append(FreeRectangle(
            free_rect.x,
            item.y + item.height,
            free_rect.width,
            free_rect.y + free_rect.height - (item.y + item.height)
        ))
    
    # Left rectangle
    if free_rect.x < item.x:
        new_rectangles.append(FreeRectangle(
            free_rect.x,
            free_rect.y,
            item.x - free_rect.x,
            free_rect.height
        ))
    
    # Bottom rectangle
    if free_rect.y < item.y:
        new_rectangles.append(FreeRectangle(
            free_rect.x,
            free_rect.y,
            free_rect.width,
            item.y - free_rect.y
        ))
    
    return new_rectangles

def find_best_free_rectangle(item: Rectangle, free_rectangles: List[FreeRectangle]) -> Tuple[FreeRectangle, bool]:
    best_rect = None
    best_area = float('inf')
    rotated = False

    for rect in free_rectangles:
        if rect.width >= item.width and rect.height >= item.height:
            area = rect.width * rect.height
            if area < best_area:
                best_rect = rect
                best_area = area
                rotated = False
        elif rect.width >= item.height and rect.height >= item.width:
            area = rect.width * rect.height
            if area < best_area:
                best_rect = rect
                best_area = area
                rotated = True

    return best_rect, rotated

def pack_items(items: List[Rectangle], bin_width: int, bin_height: int, algorithm: str = 'guillotine') -> List[Bin]:
    bins = [Bin(bin_width, bin_height)]
    
    for item in items:
        packed = False
        for bin in bins:
            best_rect, rotated = find_best_free_rectangle(item, bin.free_rectangles)
            
            if best_rect:
                if rotated:
                    item.width, item.height = item.height, item.width
                
                if algorithm == 'guillotine':
                    new_free_rects = guillotine_split(best_rect, item)
                else:  # maximal_rectangles
                    new_free_rects = maximal_rectangles_split(best_rect, item)
                
                bin.free_rectangles.remove(best_rect)
                bin.free_rectangles.extend(new_free_rects)
                bin.items.append((item, best_rect.x, best_rect.y, rotated))
                
                if algorithm == 'guillotine':
                    bin.free_rectangles = merge_rectangles(bin.free_rectangles)
                else:
                    bin.free_rectangles = [rect for rect in bin.free_rectangles if not is_fully_overlapped(rect, bin.free_rectangles)]
                
                packed = True
                break
        
        if not packed:
            new_bin = Bin(bin_width, bin_height)
            bins.append(new_bin)
            # Pack the item in the new bin
            best_rect, rotated = find_best_free_rectangle(item, new_bin.free_rectangles)
            if best_rect:
                if rotated:
                    item.width, item.height = item.height, item.width
                
                if algorithm == 'guillotine':
                    new_free_rects = guillotine_split(best_rect, item)
                else:  # maximal_rectangles
                    new_free_rects = maximal_rectangles_split(best_rect, item)
                
                new_bin.free_rectangles.remove(best_rect)
                new_bin.free_rectangles.extend(new_free_rects)
                new_bin.items.append((item, best_rect.x, best_rect.y, rotated))
                
                if algorithm == 'guillotine':
                    new_bin.free_rectangles = merge_rectangles(new_bin.free_rectangles)
                else:
                    new_bin.free_rectangles = [rect for rect in new_bin.free_rectangles if not is_fully_overlapped(rect, new_bin.free_rectangles)]
    
    return bins

def is_fully_overlapped(rect: FreeRectangle, rectangles: List[FreeRectangle]) -> bool:
    for other in rectangles:
        if other != rect and \
           other.x <= rect.x and other.y <= rect.y and \
           other.x + other.width >= rect.x + rect.width and \
           other.y + other.height >= rect.y + rect.height:
            return True
    return False

# Initialize the CNF formula
start = timeit.default_timer()
#read file
def read_file_instance(filepath):
    f = open(filepath)
    return f.read().splitlines()

def positive_range(end):
    if (end < 0):
        return []
    return range(end)

def display_solution(strip, rectangles, pos_circuits, rotation):
    # define Matplotlib figure and axis
    fig, ax = plt.subplots()
    ax = plt.gca()
    plt.title(strip)

    if len(pos_circuits) > 0:
        for i in range(len(rectangles)):
            rect = plt.Rectangle(pos_circuits[i],
                                 rectangles[i][0] if not rotation[i] else rectangles[i][1],
                                 rectangles[i][1] if not rotation[i] else rectangles[i][0],
                                 edgecolor="#333")
            ax.add_patch(rect)

    ax.set_xlim(0, strip[0])
    ax.set_ylim(0, strip[1] + 1)
    ax.set_xticks(range(strip[0] + 1))
    ax.set_yticks(range(strip[1] + 1))
    ax.set_xlabel('width')
    ax.set_ylabel('height')
    # display plot
    plt.show()

def OPP(rectangles, n, W, H):
# Define the variables
    height = H
    width = n * W
    cnf = CNF()
    variables = {}
    counter = 1

    for i in range(len(rectangles)):
        for j in range(len(rectangles)):
            if i != j:
                variables[f"lr{i + 1},{j + 1}"] = counter  # lri,rj
                counter += 1
                variables[f"ud{i + 1},{j + 1}"] = counter  # uri,rj
                counter += 1
        for e in range(width):
            variables[f"px{i + 1},{e}"] = counter  # pxi,e
            counter += 1
        for f in range(height):
            variables[f"py{i + 1},{f}"] = counter  # pyi,f
            counter += 1

    # Rotated variables
    for i in range(len(rectangles)):
        variables[f"r{i + 1}"] = counter
        counter += 1

    # Add the 2-literal axiom clauses
    for i in range(len(rectangles)):
        for e in range(width - 1):  # -1 because we're using e+1 in the clause
            cnf.append([-variables[f"px{i + 1},{e}"],
                        variables[f"px{i + 1},{e + 1}"]])
        for f in range(height - 1):  # -1 because we're using f+1 in the clause
            cnf.append([-variables[f"py{i + 1},{f}"],
                        variables[f"py{i + 1},{f + 1}"]])


    # Add non-overlapping constraints

    def non_overlapping(rotated, i, j, h1, h2, v1, v2):
        if not rotated:
            i_width = rectangles[i][0]
            i_height = rectangles[i][1]
            j_width = rectangles[j][0]
            j_height = rectangles[j][1]
            i_rotation = variables[f"r{i + 1}"]
            j_rotation = variables[f"r{j + 1}"]
        else:
            i_width = rectangles[i][1]
            i_height = rectangles[i][0]
            j_width = rectangles[j][1]
            j_height = rectangles[j][0]
            i_rotation = -variables[f"r{i + 1}"]
            j_rotation = -variables[f"r{j + 1}"]

        # Square symmertry breaking, if i is square than it cannot be rotated
        if i_width == i_height and rotated:
            i_square = True
            cnf.append([-variables[f"r{i + 1}"]])
        else:
            i_square = False

        if j_width == j_height and rotated:
            j_square = True
            cnf.append([-variables[f"r{j + 1}"]])
        else:
            j_square = False

        # lri,j v lrj,i v udi,j v udj,i
        four_literal = []
        if h1: four_literal.append(variables[f"lr{i + 1},{j + 1}"])
        if h2: four_literal.append(variables[f"lr{j + 1},{i + 1}"])
        if v1: four_literal.append(variables[f"ud{i + 1},{j + 1}"])
        if v2: four_literal.append(variables[f"ud{j + 1},{i + 1}"])

        cnf.append(four_literal + [i_rotation])
        cnf.append(four_literal + [j_rotation])

        # ¬lri, j ∨ ¬pxj, e
        if h1 and not i_square:
            for e in range(min(width, i_width)):
                    cnf.append([i_rotation,
                                -variables[f"lr{i + 1},{j + 1}"],
                                -variables[f"px{j + 1},{e}"]])
        # ¬lrj,i ∨ ¬pxi,e
        if h2 and not j_square:
            for e in range(min(width, j_width)):
                    cnf.append([j_rotation,
                                -variables[f"lr{j + 1},{i + 1}"],
                                -variables[f"px{i + 1},{e}"]])
        # ¬udi,j ∨ ¬pyj,f
        if v1 and not i_square:
            for f in range(min(height, i_height)):
                    cnf.append([i_rotation,
                                -variables[f"ud{i + 1},{j + 1}"],
                                -variables[f"py{j + 1},{f}"]])
        # ¬udj, i ∨ ¬pyi, f,
        if v2 and not j_square:
            for f in range(min(height, j_height)):
                    cnf.append([j_rotation,
                                -variables[f"ud{j + 1},{i + 1}"],
                                -variables[f"py{i + 1},{f}"]])

        for e in positive_range(width - i_width):
            # ¬lri,j ∨ ¬pxj,e+wi ∨ pxi,e
            if h1 and not i_square:
                    cnf.append([i_rotation,
                                -variables[f"lr{i + 1},{j + 1}"],
                                variables[f"px{i + 1},{e}"],
                                -variables[f"px{j + 1},{e + i_width}"]])

        for e in positive_range(width - j_width):
            # ¬lrj,i ∨ ¬pxi,e+wj ∨ pxj,e
            if h2 and not j_square:
                    cnf.append([j_rotation,
                                -variables[f"lr{j + 1},{i + 1}"],
                                variables[f"px{j + 1},{e}"],
                                -variables[f"px{i + 1},{e + j_width}"]])

        for f in positive_range(height - i_height):
            # udi,j ∨ ¬pyj,f+hi ∨ pxi,e
            if v1 and not i_square:
                    cnf.append([i_rotation,
                                -variables[f"ud{i + 1},{j + 1}"],
                                variables[f"py{i + 1},{f}"],
                                -variables[f"py{j + 1},{f + i_height}"]])
        for f in positive_range(height - j_height):
            # ¬udj,i ∨ ¬pyi,f+hj ∨ pxj,f
            if v2 and not j_square:
                    cnf.append([j_rotation,
                                -variables[f"ud{j + 1},{i + 1}"],
                                variables[f"py{j + 1},{f}"],
                                -variables[f"py{i + 1},{f + j_height}"]])

    for i in range(len(rectangles)):
        for j in range(i + 1, len(rectangles)):
           #  #Large-rectangles horizontal
            if min(rectangles[i][0], rectangles[i][1]) + min(rectangles[j][0], rectangles[j][1]) > width:
                non_overlapping(False, i, j, False, False, True, True)
                non_overlapping(True, i, j, False, False, True, True)
            # Large rectangles vertical
            elif min(rectangles[i][0], rectangles[i][1]) + min(rectangles[j][0], rectangles[j][1]) > height:
                non_overlapping(False, i, j, True, True, False, False)
                non_overlapping(True, i, j, True, True, False, False)

            # Same rectangle and is a square
            elif rectangles[i] == rectangles[j]:
                if rectangles[i][0] == rectangles[i][1]:
                    cnf.append([-variables[f"r{i + 1}"]])
                    cnf.append([-variables[f"r{j + 1}"]])
                    non_overlapping(False,i ,j, True, False, True, True)
                else:
                    non_overlapping(False,i ,j, True, False, True, True)
                    non_overlapping(True,i ,j, True, False, True, True)
           # #normal rectangles
            else:
                non_overlapping(False, i, j, True, True, True, True)
                non_overlapping(True, i, j, True, True, True, True)

    # Domain encoding to ensure every rectangle stays inside strip's boundary
    for i in range(len(rectangles)):
        if rectangles[i][0] > width: #if rectangle[i]'s width larger than strip's width, it has to be rotated
            cnf.append([variables[f"r{i + 1}"]])
        else:
            for e in range(width - rectangles[i][0], width):
                    cnf.append([variables[f"r{i + 1}"],
                                variables[f"px{i + 1},{e}"]])
        if rectangles[i][1] > height:
            cnf.append([variables[f"r{i + 1}"]])
        else:
            for f in range(height - rectangles[i][1], height):
                    cnf.append([variables[f"r{i + 1}"],
                                variables[f"py{i + 1},{f}"]])

        # Rotated
        if rectangles[i][1] > width:
            cnf.append([-variables[f"r{i + 1}"]])
        else:
            for e in range(width - rectangles[i][1], width):
                    cnf.append([-variables[f"r{i + 1}"],
                                variables[f"px{i + 1},{e}"]])
        if rectangles[i][0] > height:
            cnf.append([-variables[f"r{i + 1}"]])
        else:
            for f in range(height - rectangles[i][0], height):
                cnf.append([-variables[f"r{i + 1}"],
                            variables[f"py{i + 1},{f}"]])

    for k in range(1, n):
         for i in range(len(rectangles)):
            # Not rotated
            w = rectangles[i][0]
            cnf.append([variables[f"r{i + 1}"], variables[f"px{i + 1},{k * W - w}"],
                        -variables[f"px{i + 1},{k * W - 1}"]])
            cnf.append([variables[f"r{i + 1}"], -variables[f"px{i + 1},{k * W - w}"],
                        variables[f"px{i + 1},{k * W - 1}"]])
            
            # Rotated
            w = rectangles[i][1]
            cnf.append([-variables[f"r{i + 1}"], variables[f"px{i + 1},{k * W - w}"],
                        -variables[f"px{i + 1},{k * W - 1}"]])
            cnf.append([-variables[f"r{i + 1}"], -variables[f"px{i + 1},{k * W - w}"],
                        variables[f"px{i + 1},{k * W - 1}"]])
            
    
    with Solver(name="mc") as solver: #add all cnf to solver
        solver.append_formula(cnf)

        if solver.solve():
            pos = [[0 for i in range(2)] for j in range(len(rectangles))]
            rotation = []
            model = solver.get_model()
            print("SAT")
            result = {}
            for var in model:
                if var > 0:
                    result[list(variables.keys())[list(variables.values()).index(var)]] = True
                else:
                    result[list(variables.keys())[list(variables.values()).index(-var)]] = False

            for i in range(len(rectangles)):
                rotation.append(result[f"r{i + 1}"])
                for e in range(width - 1):
                    if result[f"px{i + 1},{e}"] == False and result[f"px{i + 1},{e + 1}"] == True:
                        pos[i][0] = e + 1
                    if e == 0 and result[f"px{i + 1},{e}"] == True:
                        pos[i][0] = 0
                for f in range(height - 1):
                    if result[f"py{i + 1},{f}"] == False and result[f"py{i + 1},{f + 1}"] == True:
                        pos[i][1] = f + 1
                    if f == 0 and result[f"py{i + 1},{f}"] == True:
                        pos[i][1] = 0
            print(pos, rotation)
            return(["sat", pos, rotation])

        else:
            print("unsat")
            return("unsat")
def interrupt(solver):
    solver.interrupt()
    
def guillotine_split(free_rect, item, rotated):
    item_width = item[1] if rotated else item[0]
    item_height = item[0] if rotated else item[1]
    remaining_width = free_rect[2] - item_width
    remaining_height = free_rect[3] - item_height
    new_rectangles = []

    if remaining_width > 0:
        new_rectangles.append((free_rect[0] + item_width, free_rect[1], remaining_width, free_rect[3]))
    
    if remaining_height > 0:
        new_rectangles.append((free_rect[0], free_rect[1] + item_height, item_width, remaining_height))

    return new_rectangles

def maximal_rectangles_split(free_rect, item, rotated):
    item_width = item[1] if rotated else item[0]
    item_height = item[0] if rotated else item[1]
    new_rectangles = []
    
    # Right rectangle
    if free_rect[0] + free_rect[2] > free_rect[0] + item_width:
        new_rectangles.append((free_rect[0] + item_width, free_rect[1], free_rect[2] - item_width, free_rect[3]))
    
    # Top rectangle
    if free_rect[1] + free_rect[3] > free_rect[1] + item_height:
        new_rectangles.append((free_rect[0], free_rect[1] + item_height, free_rect[2], free_rect[3] - item_height))
    
    return new_rectangles

def find_best_fit(item, free_rects, allow_rotation):
    best_rect = None
    best_area = float('inf')
    best_rotated = False

    for rect in free_rects:
        if rect[2] >= item[0] and rect[3] >= item[1]:
            area = rect[2] * rect[3]
            if area < best_area:
                best_rect = rect
                best_area = area
                best_rotated = False
        
        if allow_rotation and rect[2] >= item[1] and rect[3] >= item[0]:
            area = rect[2] * rect[3]
            if area < best_area:
                best_rect = rect
                best_area = area
                best_rotated = True

    return best_rect, best_rotated

def BPP(W, H, items, n, algorithm='guillotine', allow_rotation=True):
    items_area = [i[0] * i[1] for i in items]
    bin_area = W * H
    lower_bound = math.ceil(sum(items_area) / bin_area)
    
    for k in range(lower_bound, n + 1):
        print(f"Trying with {k} bins")
        result = OPP(items, k, W, H)
        if result[0] == "sat":
            print(f"Solution found with {k} bins")
            position = result[1]
            bins_used = [[i for i in range(n) if position[i][0] // W == j] for j in range(k)]
            rotation = result[2]
            
            # Apply heuristic improvement
            improved_bins = []
            for j in range(k):
                bin_items = [items[i] for i in bins_used[j]]
                bin_rotations = [rotation[i] for i in bins_used[j]]
                improved_bin = pack_bin(W, H, bin_items, algorithm, allow_rotation)
                if improved_bin is not None:
                    improved_bins.extend(improved_bin)
                else:
                    # If packing fails, use the original solution for this bin
                    original_bin = Bin(W, H, [(items[i], position[i][0] % W, position[i][1], rotation[i]) for i in bins_used[j]])
                    improved_bins.append(original_bin)
            
            # Update positions and rotations based on improved packing
            new_position = []
            new_rotation = []
            for bin_index, bin in enumerate(improved_bins):
                for item, x, y, rotated in bin.items:
                    new_position.append([x + bin_index * W, y])
                    new_rotation.append(rotated)
            
            return [improved_bins, new_position, new_rotation]
    
    return None  # No solution found

def pack_bin(W, H, items, algorithm='guillotine', allow_rotation=True):
    free_rects = [(0, 0, W, H)]
    packed_items = []

    for item in items:
        best_rect, rotated = find_best_fit(item, free_rects, allow_rotation)
        if best_rect is None:
            return None  # Item doesn't fit, bin packing failed

        item_width = item[1] if rotated else item[0]
        item_height = item[0] if rotated else item[1]
        packed_items.append((item, best_rect[0], best_rect[1], rotated))

        free_rects.remove(best_rect)
        if algorithm == 'guillotine':
            new_rects = guillotine_split(best_rect, (item_width, item_height), rotated)
        else:  # maximal_rectangles
            new_rects = maximal_rectangles_split(best_rect, (item_width, item_height), rotated)
        
        free_rects.extend(new_rects)
        free_rects = [rect for rect in free_rects if not any(
            r[0] <= rect[0] and r[1] <= rect[1] and 
            r[0] + r[2] >= rect[0] + rect[2] and 
            r[1] + r[3] >= rect[1] + rect[3] 
            for r in free_rects if r != rect
        )]

    return [Bin(W, H, packed_items)]

class Bin:
    def __init__(self, width, height, items):
        self.width = width
        self.height = height
        self.items = items  # List of (item, x, y, rotated) tuples

def print_solution(bpp_result):
    if bpp_result is None:
        print("No solution found")
        return

    bins = bpp_result[0]
    pos = bpp_result[1]
    rotation = bpp_result[2]
    print("Positions:", pos)
    print("Rotations:", rotation)
    
    for i, bin in enumerate(bins):
        print(f"Bin {i + 1} contains items:")
        bin_items = []
        bin_positions = []
        bin_rotations = []
        
        for j, (item, x, y, rotated) in enumerate(bin.items):
            if rotated:
                print(f"Rotated item {j + 1} {item} at position ({x}, {y})")
            else:
                print(f"Item {j + 1} {item} at position ({x}, {y})")
            
            bin_items.append(item)
            bin_positions.append((x, y))
            bin_rotations.append(rotated)
        
        display_solution((bin.width, bin.height), bin_items, bin_positions, bin_rotations)

# Make sure to define items before calling this function
# items should be a list of (width, height) tuples for each item

# Example usage:
# bpp_result = BPP(W, H, items, n, algorithm='guillotine', allow_rotation=True)
# print_solution(bpp_result)

input = read_file_instance("input_data/ins-2.txt")
n = int(input[0])
bin_size = input[1].split()
W = int(bin_size[0])
H = int(bin_size[1])
items = [[int(val) for val in i.split()] for i in input[2:]]

bpp_result = BPP(W, H, items, n, algorithm='guillotine', allow_rotation=True)
print_solution(bpp_result)
stop = timeit.default_timer()
print("Time:", stop - start)