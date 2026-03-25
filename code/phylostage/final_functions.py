# ----------------------------------------------------------------------------------------------------
# all function names (to help ctrl+f) (sorted by section)

"""
simulations
    simulate_tumor_trees
    get_trajectories
    print_trajectory
    print_trajectory_1
    get_all_trees
    tumor_tree_to_formatted_str
    tumor_trees_to_formatted_trees_file
    run_MASTRO
    show_MASTRO_results
    run_simulation_and_save_result
dataset_summaries
    get_raw_str_trees
    get_alterations_and_relationships_dict
    show_summary
    get_alteration_frequencies
    get_alterations_with_min_frequency
significance testing
    permute_patient_trees
    get_stage_score
    get_strict_score
    get_partial_score
    empirical_p1_value
    p2_value
    get_effect_size
matrices
	make_patient_tree_clone_alteration_matrix
	make_three_matrices
misc
	make_CBNPM
	CBNPM_tumor_tree_consistency
	get_strictly_consistent_patients
	show_strictly_consistent_patients
	prune_CBNPM
	get_patients_with_enough_alterations_dependencies_depth
    trees_lists_equals
    get_tree_stage
    get_patient_stage
    get_TRACERx_patient_trees_lists
    TRACERx_patient_number_to_name
main problem
	problem_1_1
	problem_1_1_display_results
	problem_1_1_make_ilp
	problem_1_1_get_solution
	problem_1_1_validate_solution
	problem_1_1_display_validation_result
    run_problem_1_1_and_save_result
    run_problem_1_1_full_significance_testing
from pm.py
	tumor_tree_to_branch_list
	get_width_from_font_size
	show_tree
	get_file_path
	get_all_alterations
	get_relationships_matrix
	relationship_type
from pm2.py
	make_tree1
	get_mutations_from_tree
	get_patient_trees_lists
    show_patient_trees
    make_collapsed_tree1
"""

# ----------------------------------------------------------------------------------------------------
# imports (may not use all)

import re
import numpy as np
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math
import itertools
import gurobipy as gp
from gurobipy import GRB
import time
import os
import textwrap
import copy
import sys
import io
import random
import subprocess
import shutil
import pandas as pd
from IPython.display import display
import ast

# ----------------------------------------------------------------------------------------------------
# simulations

def simulate_tumor_trees(seed, num_trees, pathways, alterations, tree_follows_PM_prob, node_follows_PM_prob, topology_type, topologies=None, branch_prob=None, progression_prob=None):
    """
    seed is the random seed
    num_trees is how many tumor trees to simulate from the progression model (though, if topology_type is "given", this currently will be ignored and it'll make as many trees as there are topologies
    the oprogression model is implicitly defined by 'pathways' (list of lists), for now just a linear chain (pathways[0] -> pathways[1] -> ...)
    alterations is the full list of alterations from which to choose random ones if tree not following the progression model or similar
    tree_follows_PM_prob is probability that the tree will follow the PM
    node_follows_PM_prob is probability that a node will follow the PM, for a tree that does
    topology_type is "given" or "random"- if given, it's based on topologies; if random, it's based on branch_prob
    if topology_type is "given", topologies should be passed in, and be the same length as num_trees
    if topology_type is "random", a branch_prob should be passed in, which is the probability of branching out, and a progression_prob, which is probability of same number of levels as number of pathways
    the topologies must be trees
    returns a list of simulated trees
    """

    p1 = tree_follows_PM_prob
    p2 = node_follows_PM_prob
    p3 = branch_prob
    p4 = progression_prob

    random.seed(seed)
    
    def get_num_alterations(node):
        if (node == " "):
            return 0
        elif (isinstance(node, str)):
            return 1
        else: # tuple
            return len(node)

    def simulate_tumor_tree(topology=None):
        follows_PM = random.random() < p1
        
        all_alterations = set(alterations)
        alterations_in_pm = set(alteration for pathway in pathways for alteration in pathway)
        used_alterations = set()

        tree = nx.DiGraph()

        if (topology is not None): # given topology
            queue = [(" ", " ", 0)] # (curr_node_simulated, curr_node_topology, curr_node_next_pathway_index) 
        
            while (len(queue) > 0):
                curr_node_simulated, curr_node_topology, curr_node_next_pathway_index = queue.pop(0)
                
                for next_node_topology in sorted(topology.successors(curr_node_topology)):
                    num_alterations_in_node = get_num_alterations(next_node_topology)
                    chosen_alterations = []
                    next_pathway_index = curr_node_next_pathway_index
                    
                    if (follows_PM):
                        if (random.random() < p2): # tree follows pm, node follows pm- we will select from the pathway(s) if possible

                            for _ in range(num_alterations_in_node): # preserving the number of alterations in each node
                                if (next_pathway_index < len(pathways)): # more pathways left- we will select from the current pathway if possible
                                    
                                    possible_alterations = sorted(set(pathways[next_pathway_index]) - used_alterations) # mutations in current pathway that haven't been used yet

                                    if (len(possible_alterations) == 0): # all alterations in the pathway were used up in other branches
                                        possible_alterations = sorted(all_alterations - alterations_in_pm - used_alterations) # so we select from ones not in the pm
                                    else:
                                        next_pathway_index += 1
                                
                                else: # no more pathways left- we will select from alterations not in the pm
                                    possible_alterations = sorted(all_alterations - alterations_in_pm - used_alterations)

                                chosen_alterations.append(random.choice(possible_alterations))
                                used_alterations.update(chosen_alterations)

                        else: # tree follows pm, but node doesn't- we will select from alterations not in the pm
                            possible_alterations = sorted(all_alterations - alterations_in_pm - used_alterations)
                            chosen_alterations.extend(random.sample(possible_alterations, num_alterations_in_node))
                            
                    else: # tree doesn't follow pm, we will select from any alterations
                        possible_alterations = sorted(all_alterations - used_alterations)
                        chosen_alterations.extend(random.sample(possible_alterations, num_alterations_in_node))
                    
                    if (isinstance(next_node_topology, tuple)):
                        next_node_simulated = tuple(sorted(chosen_alterations))
                    else: # str
                        assert len(chosen_alterations) == 1
                        next_node_simulated = chosen_alterations[0]
                    
                    used_alterations.update(chosen_alterations)
                    tree.add_edge(curr_node_simulated, next_node_simulated)
                    queue.append((next_node_simulated, next_node_topology, next_pathway_index))
                                    
        else: # random topology
            raise Exception("random topology not implemented rn (old one needed to be fixed)")

        return tree

    if topology_type == "given":
        return [simulate_tumor_tree(topology) for topology in topologies]
    else:
        return [simulate_tumor_tree(None) for _ in range(num_trees)]

def get_trajectories(pathways):
    """gets all trajectories (formatted as a CBNPM for show_strictly_and_partially_consistent_patients) from a linear chain PM implicitly defined by the given pathways"""

    trajectories = list(itertools.product(*pathways))
    return [make_CBNPM({i: [alteration] for i, alteration in enumerate(trajectory, start=1)}, [(1, 2), (2, 3)]) for trajectory in trajectories]

def print_trajectory(trajectory):
    """prints a trajectory (formatted as CBNPM) as simple chain like alteration -> alteration -> ..."""

    curr_node = [node for node in trajectory.nodes() if trajectory.in_degree(node) == 0][0]
    print(curr_node[0], end="")
    while (trajectory.out_degree(curr_node) != 0):
        curr_node = list(trajectory.successors(curr_node))[0]
        print(f" -> {curr_node[0]}", end="")
    print()

def print_trajectory_1(trajectory):
    """prints a trajectory (an nx graph w/ str nodes that is a simple linear chain) as like alteration -> alteration -> ..."""

    curr_node = [node for node in trajectory.nodes() if trajectory.in_degree(node) == 0][0]
    path = [curr_node]
    while trajectory.out_degree(curr_node) > 0:
        curr_node = list(trajectory.successors(curr_node))[0]
        path.append(curr_node)
    print(" -> ".join(node for node in path))

def get_all_trees(path_to_formatted_trees):
    
    def make_tree(relationships):
        ancestral_relationships = [tuple(re.split('-.-', relationship)) for relationship in relationships if relationship_type(relationship) == "ancestral"]
        clustered_relationships = [tuple(re.split('-.-', relationship)) for relationship in relationships if relationship_type(relationship) == "clustered"]
        
        G = nx.DiGraph()
        
        for relationship in ancestral_relationships:
            G.add_edge(relationship[0], relationship[1])
    
        G = nx.transitive_reduction(G)
    
        genes_set = set(gene for pair in relationships for gene in tuple(re.split('-.-', pair)))
        for gene in genes_set:
            if (not G.has_node(gene) or G.in_degree(gene) == 0):
                G.add_edge(" ", gene)
    
        clusters = [] # list of lists
        for (one, two) in clustered_relationships:
            
            one_index, two_index = -1, -1 # will be -1 if in no cluster
            for i, cluster in enumerate(clusters):
                if one in cluster:
                    one_index = i
                if two in cluster:
                    two_index = i
    
            if (one_index == -1 and two_index == -1):
                clusters.append([one, two])
            elif (one_index != -1 and two_index == -1):
                clusters[one_index].append(two)
            elif (two_index != -1 and one_index == -1):
                clusters[two_index].append(one)
            elif (one_index != two_index): # they are both in clusters but in different, so they have to be merged (this won't happen w/ trees-aml data but could w/ other)
                 clusters[one_index].extend(clusters[two_index])
                 del clusters[two_index]
    
        for cluster in clusters:
            G.add_node(tuple(cluster)) 
            for old_node in cluster:
                G = nx.contracted_nodes(G, tuple(cluster), old_node)
    
        return G
        
    trees = []
    
    relationships_matrix = get_relationships_matrix(path_to_formatted_trees)

    for relationships in relationships_matrix:
        trees.append(make_tree(relationships))

    return trees

def tumor_tree_to_formatted_str(tree):
    ret_str = ""
    
    alterations = set(get_mutations_from_tree(tree)) - {" "}
    mapping = {alteration: node for node in tree for alteration in (node if isinstance(node, tuple) else [node])}

    transitive_closure = nx.transitive_closure(tree)
    
    for a, b in itertools.combinations(alterations, 2):
        if (transitive_closure.has_edge(mapping[a], mapping[b])):
            ret_str += f"{a}->-{b} "
        elif (transitive_closure.has_edge(mapping[b], mapping[a])):
            ret_str += f"{b}->-{a} "
        elif (mapping[a] == mapping[b]):
            ret_str += f"{a}-?-{b} " if a < b else f"{b}-?-{a} "
        else:
            ret_str += f"{a}-/-{b} " if a < b else f"{b}-/-{a} "

    return ret_str

def tumor_trees_to_formatted_trees_file(trees, file_path):
    with open(file_path, "w") as f:
        for tree in trees:
            f.write(tumor_tree_to_formatted_str(tree) + "\n")

    # ensuring it was done right
    recovered_trees = get_all_trees(file_path)
    try:
        assert len(trees) == len(recovered_trees) and trees_lists_equals(trees, recovered_trees)
    except:
        print("the tumor trees were not correctly recovered from the MASTRO-formatted trees file, returning the recovered trees", file=sys.stderr)
        return recovered_trees

def run_MASTRO(simulated_trees, MASTRO_format_trees_file_path):
    start_all = time.perf_counter()
    
    results = {}

    MASTRO_path = "../MASTRO/MASTRO"
    simulation_file_name = os.path.basename(MASTRO_format_trees_file_path)

    original_files = set(os.listdir(MASTRO_path))
    
    if ((ret := tumor_trees_to_formatted_trees_file(simulated_trees, MASTRO_format_trees_file_path)) != None): return ret # the tumor trees weren't correctly put in MASTRO format?

    shutil.copy(MASTRO_format_trees_file_path, MASTRO_path)

    start_MASTRO = time.perf_counter()
    result = subprocess.run(["python3", "run_MASTRO.py", "-g", simulation_file_name, "-p", "1", "-s", "2"], cwd=MASTRO_path, capture_output=True, text=True) # -p 1 for permutation
    results['time_took_MASTRO'] = time.perf_counter() - start_MASTRO
    
    results["stdout"] = result.stdout
    results["stderr"] = result.stderr
    
    lines = result.stdout.strip().split("\n")
    results["min_p_value"] = float(lines[0].split("min pvalue ")[1])
    min_traj_info = lines[1] # edges, support, and tree indexes observed in

    pattern = r"(\[.*?\])\s+(\d+)\s+(\[.*?\])"
    match = re.search(pattern, min_traj_info)
    
    if match:
        results["edges"] = ast.literal_eval(match.group(1))
        trajectory = nx.DiGraph(results["edges"])
        trajectory.remove_node("g")
        trajectory = nx.transitive_reduction(trajectory)
        results["trajectory"] = trajectory

        results["support"] = int(match.group(2))
        results["tree_indexes_observed_in"] = [i + 1 for i in ast.literal_eval(match.group(3))]
    else:
        raise Exception(f"Failed to parse min trajectory info: {min_traj_info}")

    results_file = os.path.join(MASTRO_path, simulation_file_name.removesuffix(".txt") + "_final.txt")
    results["results_df"] = pd.read_csv(results_file, sep=";")

    cleanup_dir = os.path.join(MASTRO_path, "cleanup")
    os.makedirs(cleanup_dir, exist_ok=True)
    current_files = set(os.listdir(MASTRO_path))
    for file in current_files - original_files - {"cleanup"}:
        shutil.move(os.path.join(MASTRO_path, file), os.path.join(cleanup_dir, file))

    results['time_took_all'] = time.perf_counter() - start_all

    return results

def show_MASTRO_results(results, csv_save_path=None):
    print(f"time took to run run_MASTRO function: {results["time_took_all"]}")
    print(f"time took to run MASTRO itself: {results["time_took_MASTRO"]}")
    print(f"\nsupport of trajectory with min p-value: {results["support"]}")
    print(f"p-value: {results["min_p_value"]}")
    print(f"indexes of trees the trajectory was observed in: {results["tree_indexes_observed_in"]}")
    print(f"\nraw MASTRO output (mainly to ensure it was parsed correctly):\n\nSTDOUT: {results["stdout"]}\nSTDERR: {results["stderr"]}")

    if (csv_save_path is not None):
        print("\ntrajectory with min p-value:")
        print_trajectory_1(results["trajectory"])

        print(f"\nfull results df in {os.path.basename(csv_save_path)}")
        results["results_df"].to_csv(csv_save_path, index=False)

    else:
        print("\ntrajectory with min p-value:")
        show_tree(results["trajectory"])
    
        print("\nfull results df:")
        display(results["results_df"])

def run_simulation_and_save_result(p1, p2, seed, save_folder_path, make_trees_pdf=False):
    """this assumes you are using pathways = [['IDH1', 'IDH2'], ['SRSF2'], ['NRAS', 'KRAS']] and AML topologies
    writes results to save_folder_path, and also
    returns list with [p1, p2, seed, our ilp w/ max 2 alterations alteration recall, our ilp w/ max 2 alterations alteration precision, our ilp w/ max 2 alterations pathway accuracy, our ilp w/ max 2 alterations dependency accuracy, our ilp w/ no max alterations alteration recall, our ilp w/ no max alterations alteration precision, our ilp w/ no max alterations pathway accuracy, our ilp w/ no max alterations dependency accuracy]"""

    # setup

    os.makedirs(save_folder_path, exist_ok=True)
    log_path = os.path.join(save_folder_path, "log.txt")

    old_stdout = sys.stdout
    sys.stdout = open(log_path, "w")
    try:

        ILP_max_2_alt_PPM_path = os.path.join(save_folder_path, "ILP_max_2_alt_PPM.png")
        ILP_no_max_alt_PPM_path = os.path.join(save_folder_path, "ILP_no_max_alt_PPM.png")
        MASTRO_results_path = os.path.join(save_folder_path, "MASTRO_results.csv")
        simulated_trees_MASTRO_format_path = os.path.join(save_folder_path, "simulated_trees_MASTRO_format_path.txt")
        if (make_trees_pdf): trees_pdf_path = os.path.join(save_folder_path, "trees.png")
    
        # simulate
        
        topologies = [trees[0] for trees in get_patient_trees_lists("AML")]
        alterations = set(get_all_alterations("AML")) - {" "}
        pathways = [['IDH1', 'IDH2'], ['SRSF2'], ['NRAS', 'KRAS']]
        
        simulated_trees = simulate_tumor_trees(seed=seed, num_trees=len(topologies), pathways=pathways, alterations=alterations, tree_follows_PM_prob=p1, node_follows_PM_prob=p2, topology_type="given", topologies=topologies, branch_prob=None, progression_prob=None)
    
        if (make_trees_pdf): show_patient_trees([[tree] for tree in simulated_trees], save_path=trees_pdf_path)
    
        # see how many strictly and partially consistent with the CBNPM, and with each trajectory
        print("----------------------------------------------------------------------------------------------------\nSee how many strictly and partially consistent with the CBNPM, and with each trajectory\n")
    
        print(f"Original linear progression model with pathways: {pathways}")
        print()
        show_strictly_and_partially_consistent_patients(make_CBNPM({i: pathway for i, pathway in enumerate(pathways, start=1)}, [(k, k+1) for k in range(1, len(pathways))]), [[tree] for tree in simulated_trees])
        print()
        
        trajectories = get_trajectories(pathways)
        
        for i, trajectory in enumerate(trajectories, start=1):
            print(f"Trajectory {i}: ", end="")
            print_trajectory(trajectory)
            print()
            show_strictly_and_partially_consistent_patients(trajectory, [[tree] for tree in simulated_trees])
            print()
    
        # run our ilp on simulated trees to get pathway progression model
        print("----------------------------------------------------------------------------------------------------\nRun our ilp on simulated trees to get pathway progression model\n")
    
        print("With a max alterations per pathway of 2\n")
    
        patient_trees = [[tree] for tree in simulated_trees]
        alterations = alterations
        num_levels = len(pathways)
        num_pathways = len(pathways)
        num_dependencies = len(pathways) - 1
        max_solutions = 1
        max_alterations_per_pathway = 2
        
        ilp_results_simulated_max_2_alt = problem_1_1(patient_trees, alterations, num_pathways, num_dependencies, num_levels, max_solutions=max_solutions, max_alterations_per_pathway=max_alterations_per_pathway)
    
        print()
        problem_1_1_display_results(ilp_results_simulated_max_2_alt, save_path=ILP_max_2_alt_PPM_path)
    
        print("With no max alterations per pathway\n")
    
        max_alterations_per_pathway = None
    
        ilp_results_simulated_no_max_alt = problem_1_1(patient_trees, alterations, num_pathways, num_dependencies, num_levels, max_solutions=max_solutions, max_alterations_per_pathway=max_alterations_per_pathway)
    
        print()
        problem_1_1_display_results(ilp_results_simulated_no_max_alt, save_path=ILP_no_max_alt_PPM_path)
    
        # run MASTRO on simulated trees
        print("----------------------------------------------------------------------------------------------------\nRun MASTRO on simulated trees\n")
    
        MASTRO_results = run_MASTRO(simulated_trees, simulated_trees_MASTRO_format_path)
        if (not isinstance(MASTRO_results, dict)): # the tumor trees weren't correctly put in MASTRO format?
            print("ERROR: the tumor trees weren't correctly put in MASTRO format?")
            return MASTRO_results
    
        show_MASTRO_results(MASTRO_results, csv_save_path=MASTRO_results_path)

        # calculate metrics
    
        def node_to_sorted_tuple(node):
            return tuple(sorted(node if isinstance(node, tuple) else [node])) # actually pm nodes might always be tuples but just in case
    
        true_pathways = set(tuple(sorted(pathway)) for pathway in pathways) # true/predicted pathways will be a set of sorted tuples
        true_dependencies = set((tuple(sorted(pathways[i])), tuple(sorted(pathways[i+1]))) for i in range(len(pathways) - 1)) # true/predicted dependencies will be a set of 2-tuples of sorted tuples
    
        predicted_PM_max_2_alt = ilp_results_simulated_max_2_alt['CBNPM_list'][0]
        predicted_pathways_max_2_alt = set(node_to_sorted_tuple(node) for node in predicted_PM_max_2_alt.nodes())
        predicted_dependencies_max_2_alt = set((node_to_sorted_tuple(parent), node_to_sorted_tuple(child)) for parent, child in predicted_PM_max_2_alt.edges())
    
        true_alterations = set(alteration for pathway in true_pathways for alteration in pathway)
        predicted_alterations_max_2_alt = set(alteration for pathway in predicted_pathways_max_2_alt for alteration in pathway)
        ours_alteration_recall_max_2_alt = len(true_alterations & predicted_alterations_max_2_alt) / len(true_alterations)
        ours_alteration_precision_max_2_alt = len(true_alterations & predicted_alterations_max_2_alt) / len(predicted_alterations_max_2_alt)
    
        ours_pathway_accuracy_max_2_alt = len(true_pathways & predicted_pathways_max_2_alt) / len(true_pathways) # accuracy = recall = precision since len(true) = len(predicted), same for dependency accuracy
        ours_dependency_accuracy_max_2_alt = len(true_dependencies & predicted_dependencies_max_2_alt) / len(true_dependencies)
    
        predicted_PM_no_max_alt = ilp_results_simulated_no_max_alt['CBNPM_list'][0] 
        predicted_pathways_no_max_alt = set(node_to_sorted_tuple(node) for node in predicted_PM_no_max_alt.nodes())
        predicted_dependencies_no_max_alt = set((node_to_sorted_tuple(parent), node_to_sorted_tuple(child)) for parent, child in predicted_PM_no_max_alt.edges())
    
        true_alterations = set(alteration for pathway in true_pathways for alteration in pathway)
        predicted_alterations_no_max_alt = set(alteration for pathway in predicted_pathways_no_max_alt for alteration in pathway)
        ours_alteration_recall_no_max_alt = len(true_alterations & predicted_alterations_no_max_alt) / len(true_alterations)
        ours_alteration_precision_no_max_alt = len(true_alterations & predicted_alterations_no_max_alt) / len(predicted_alterations_no_max_alt)
    
        ours_pathway_accuracy_no_max_alt = len(true_pathways & predicted_pathways_no_max_alt) / len(true_pathways) # accuracy = recall = precision since len(true) = len(predicted), same for dependency accuracy
        ours_dependency_accuracy_no_max_alt = len(true_dependencies & predicted_dependencies_no_max_alt) / len(true_dependencies)
            
        return [p1, p2, seed, ours_alteration_recall_max_2_alt, ours_alteration_precision_max_2_alt, ours_pathway_accuracy_max_2_alt, ours_dependency_accuracy_max_2_alt, ours_alteration_recall_no_max_alt, ours_alteration_precision_no_max_alt, ours_pathway_accuracy_no_max_alt, ours_dependency_accuracy_no_max_alt]

    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

# ----------------------------------------------------------------------------------------------------
# dataset_summaries 

def get_raw_str_trees(dataset):
    tree_file = get_file_path(dataset)
    file = open(tree_file, "r")
    return file.readlines()

def get_alterations_and_relationships_dict(raw_str_trees):
    relationships_dict = {"ancestral": [], "incomparable": [], "clustered": []}
    
    alterations = []
    for tree in raw_str_trees:
        relationships = tree.split()
        for relationship in relationships:
            r_type = relationship_type(relationship)
            rel_tuple = tuple(re.split('-.-', relationship))
            relationships_dict[r_type].append(rel_tuple)
            alterations.extend(rel_tuple)
    alterations = list(set(alterations))

    return alterations, relationships_dict

def show_summary(dataset):
    print(f"dataset: {dataset}")
    
    patient_trees = get_patient_trees_lists(dataset)

    print(f"there are {len(patient_trees)} patients")
    num_trees_for_each_patient = [len(trees) for trees in patient_trees]
    num_trees = sum(num_trees for num_trees in num_trees_for_each_patient)
    print(f"there are {num_trees} trees")
    print(f"the median number of trees per patient is {np.median(num_trees_for_each_patient)}")

    if (dataset == "TRACERx"):
        alteration_frequencies = get_alteration_frequencies(patient_trees)
        alterations = list(alteration_frequencies.keys())

        print(f"there are {len(alterations)} unique alterations")
        print(f"alterations: {alterations}")

        sorted_alterations = alteration_frequencies.most_common()
        print(f"alteration frequency: {sorted_alterations}")

        return

    raw_str_trees = get_raw_str_trees(dataset) 
    alterations, relationships_dict = get_alterations_and_relationships_dict(raw_str_trees)

    print(f"there are {len(alterations)} unique alterations")
    print(f"alterations: {alterations}")
    
    print(f"there are {len(relationships_dict['ancestral'])} ancestral relationships")
    print(f"there are {len(relationships_dict['incomparable'])} incomparable relationships")
    print(f"there are {len(relationships_dict['clustered'])} clustered relationships")

    alteration_frequencies = get_alteration_frequencies(patient_trees)
    sorted_alterations = alteration_frequencies.most_common()
    print(f"alteration frequency: {sorted_alterations}")

    # [more]

def get_alteration_frequencies(patient_trees):
    alteration_frequency = Counter()
    for patient in patient_trees:
        alterations = set()
        for tree in patient:
            alterations.update(get_mutations_from_tree(tree))
        alteration_frequency.update(alterations - {" "})
    sorted_alterations = alteration_frequency.most_common()

    return alteration_frequency

def get_alterations_with_min_frequency(patient_trees, min_frequency):

    alteration_frequency = get_alteration_frequencies(patient_trees)
    return [alteration for alteration, frequency in alteration_frequency.items() if frequency >= min_frequency]

# ----------------------------------------------------------------------------------------------------
# significance testing

def permute_patient_trees(patient_trees, seed=None):
    def permute_tree(tree):
        alterations = []
        for node in tree.nodes():
            if isinstance(node, tuple):
                alterations.extend(node)
            elif node != " ":
                alterations.append(node)
        
        random.shuffle(alterations)
        
        mapping = {}
        idx = 0
        for node in tree.nodes():
            if node == " ":
                mapping[node] = node
            elif isinstance(node, tuple):
                mapping[node] = tuple(alterations[idx:idx+len(node)])
                idx += len(node)
            else:
                mapping[node] = alterations[idx]
                idx += 1
        
        return nx.relabel_nodes(tree, mapping)

    if (seed is not None): random.seed(seed)

    return [[permute_tree(tree) for tree in patient] for patient in patient_trees]

def get_stage_score(progression_model, patient_trees):
    """the sum of stages for each partially consistent patient (not consistent get 0)"""

    return sum((stage if (stage := get_patient_stage(progression_model, patient)) != -1 else 0) for patient in patient_trees)

def get_strict_score(progression_model, patient_trees):
    """the number of strictly consistent patients"""

    return sum(any(CBNPM_tumor_tree_consistency(progression_model, tree, "strict") for tree in patient) for patient in patient_trees)

def get_partial_score(progression_model, patient_trees):
    """the number of partially consistent patients"""

    return sum(any(CBNPM_tumor_tree_consistency(progression_model, tree, "partial") for tree in patient) for patient in patient_trees)
    
def empirical_p1_value(score_function, null_distr_function, progression_model, real_patient_trees, num_null_datasets, seed=None):
    """finds empirical p_1-value for the progression model P which is the proportion of null datasets that have a >= score for P as the real dataset does
    score_function is a function that takes in the progression model and a dataset (patient_trees, which is list of lists), and returns a real-values score
    null_distr_function takes in a dataset and returns a new null dataset
    progression_model is a DAG on pathways, doesn't have a root node
    real_patient_trees is a list of lists (one for each patient) of tumor trees (which is tree on alterations or sets of alterations, has root node " ")
    num_null_datasets is the number of null datasets to generate and get the score for 
    seed is the random seed to be set
    returns (1) the p_1-value, (2) number of null datasets with >= score for P as the real dataset does, (3) real score, (4) a list of the scores for each null dataset"""

    if (seed is not None): random.seed(seed)
    
    real_score = score_function(progression_model, real_patient_trees)
    null_scores = [score_function(progression_model, null_distr_function(real_patient_trees)) for _ in range(num_null_datasets)]

    num_null_as_good = sum(null_score >= real_score for null_score in null_scores)
    return num_null_as_good / num_null_datasets, num_null_as_good, real_score, null_scores

def p2_value(score_function, null_distr_function, real_patient_trees, num_null_datasets, num_runs, alterations, num_pathways, num_dependencies, num_levels, max_alterations_per_pathway, seed=None):
    """finds the p_2-value for the real dataset which is the proportion of null datasets that have a <= p_1-value as the real dataset
    uses empirical_p1_value() and problem_1_1()
    score_function, null_distr_function, real_patient_trees, num_null_datasets are all passed to empirical_p1_value, as well a progression_model made for each run
    num_runs is how many total null datasets to run the ilp on and find the p_1-value for
    seed is the random seed to be set
    alterations, num_pathways, num_dependencies, num_levels, and max_alterations_per_pathway are the parameters to be passed to problem_1_1() (besides the patient_trees)
    returns (1) the p_2-value, (2) number of null datasets with <= p_1-value as the real dataset, (3) real (real_p_value, num_null_as_good, real_score, null_score) tuple from empirical_p1_value(), (4) a list of the (real_p_value, [etc]) tuples (one for each null dataset)"""
    
    if (seed is not None): random.seed(seed)

    problem_1_1_real_result = problem_1_1(real_patient_trees, alterations, num_pathways, num_dependencies, num_levels, max_solutions=1, max_alterations_per_pathway=max_alterations_per_pathway)
    real_progression_model = problem_1_1_real_result['CBNPM_list'][0]
    real_p1_value_etc = empirical_p1_value(score_function, null_distr_function, real_progression_model, real_patient_trees, num_null_datasets)
    null_p1_values_etc = []

    for _ in range(num_runs):
        null_dataset = null_distr_function(real_patient_trees)
        problem_1_1_null_result = problem_1_1(null_dataset, alterations, num_pathways, num_dependencies, num_levels, max_solutions=1, max_alterations_per_pathway=max_alterations_per_pathway)
        null_progression_model = problem_1_1_null_result['CBNPM_list'][0]
        null_p1_values_etc.append(empirical_p1_value(score_function, null_distr_function, null_progression_model, null_dataset, num_null_datasets)) # real_p_value, num_null_as_good, real_score, null_score

    num_null_as_good = sum(null_p1_value_etc[0] <= real_p1_value_etc[0] for null_p1_value_etc in null_p1_values_etc)
    return num_null_as_good / num_runs, num_null_as_good, real_p1_value_etc, null_p1_values_etc

def get_effect_size(real_score, null_scores):
    return (real_score - np.mean(null_scores)) / np.std(null_scores)

# ----------------------------------------------------------------------------------------------------
# matrices 

def make_patient_tree_clone_alteration_matrix(patient_trees, alterations):
    """
    Constructs a 4D matrix M such that M[p][t][i][m] is 1 if patient p's t'th tree's i'th clone has alteration m
    The order of patients is the same as in patient_trees, the orders of the other indexes are arbitrary
    
    Params:
        patient_trees (list of lists of DiGraph's (each node is str or [tuple of strs]): 
            A list of patients [p_1, ..., p_n]. Each patient is a list of NetworkX DiGraph objects,
            where each one is a tree, and each node in a tree is a single alteration (str) or a cluster (tuple of strs); (the root node is " ")
        alterations (set of strs): A set of alterations

    Returns:
        M (list of lists of lists of dicts (each with str keys and int (1 or 0) values)): the matrix
    """

    M = []
    
    for patient in patient_trees:
        patient_matrix = []
        
        for tree in patient:
            tree_matrix = []

            for node in tree.nodes():
                clone_dict = {}
                
                path = nx.shortest_path(tree, source=" ", target=node)
                alterations_in_clone = [alteration for path_node in path for alteration in (path_node if isinstance(path_node, tuple) else [path_node]) if alteration != " "]

                for alteration in alterations:
                    clone_dict[alteration] = 1 if alteration in alterations_in_clone else 0

                tree_matrix.append(clone_dict)

            patient_matrix.append(tree_matrix)

        M.append(patient_matrix)

    return M

def make_three_matrices(patient_trees, alterations):
    """
    Using the patient_trees, for the given set of alterations, makes the clone_alteration_matrix, tree_clone_matrix, and patient_tree_matrix
    The order of patients (rows) of patient_tree_matrix is the same as in patient_trees
    
    Params:
        patient_trees (list of lists of DiGraph's (each node is str or [tuple of strs]): 
            A list of patients [p_1, ..., p_n]. Each patient is a list of NetworkX DiGraph objects,
            where each one is a tree, and each node in a tree is a single alteration (str) or a cluster (tuple of strs); (the root node is " ")
        alterations (set of strs): A set of alterations

    Returns:
        clone_alteration_matrix (list of dicts (each with str keys and int (1 or 0) values)): which clones (1st index) have which alterations (2nd index)
        tree_clone_matrix (list of lists of ints (1 or 0)): which trees (rows) have which clones (cols)
        patient_tree_matrix (list of lists of ints (1 or 0)): which patients (rows) have which trees (cols)
    """

    M = make_patient_tree_clone_alteration_matrix(patient_trees, alterations) # list of lists of lists of dicts

    M1 = []
    for patient in M:
        for tree in patient:
            for clone in tree:
                if clone not in M1:
                    M1.append(clone)

    M2 = []
    for patient in M:
        for tree in patient:
            tree_clone_entry = [1 if clone in tree else 0 for clone in M1]
            if tree_clone_entry not in M2:
                M2.append(tree_clone_entry)

    M3 = []
    for patient in M:
        curr_patient_trees_formatted = [ [1 if clone in tree else 0 for clone in M1] for tree in patient]
        patient_tree_entry = [1 if tree in curr_patient_trees_formatted else 0 for tree in M2]
        M3.append(patient_tree_entry) # we add all patients, even in case of duplicates

    return M1, M2, M3

# ----------------------------------------------------------------------------------------------------
# misc

def make_CBNPM(pathways, edges): # conjunctive bayesian network progression model
    """pathways is dict of lists like {1: pathway 1, etc}, edges is list of tuples of (parent, child) where each is the number of a pathway"""
    
    CBNPM = nx.DiGraph() 
    CBNPM.add_nodes_from((tuple(pathway) for pathway in pathways.values())) # # in case there's a lone pathway, it still needs to be in the graph
    CBNPM.add_edges_from(((tuple(pathways[k]), tuple(pathways[kk])) for k, kk in edges))
    return CBNPM

def CBNPM_tumor_tree_consistency(CBNPM, tree, consistency_type):
    # just, every clone must be consistent (i.e., cannot have two mutations in same pathway, and cannot have an alteration that's dependent on an alteration it doesn't have), and has at least one every pathway
    # this basically just duplicates the logic of the ilp but we can double check visually to make sure they make sense
    # consistency_type is "strict" or "partial"

    pathways = list(CBNPM.nodes())
    pathway_to_index = {pathway: i for i, pathway in enumerate(pathways)}
    tree_has_pathway = [False for pathway in pathways]
    alteration_to_pathway_index = {alteration: pathway_to_index[pathway] for pathway in pathways for alteration in pathway}
    alterations = alteration_to_pathway_index.keys()
    clone_alteration_matrix = make_patient_tree_clone_alteration_matrix([[tree]], alterations)[0][0]

    for clone in clone_alteration_matrix:
        alterations_present = [alteration for alteration in alterations if clone[alteration] == 1]
        clone_has_pathway = [False for pathway in pathways]
        
        for alteration in alterations_present:
            pathway = alteration_to_pathway_index[alteration]
            if clone_has_pathway[pathway]: return False
            else: clone_has_pathway[pathway] = True
            tree_has_pathway[pathway] = True

        for dependency in CBNPM.edges():
            first = pathway_to_index[dependency[0]]
            second = pathway_to_index[dependency[1]]
            if (clone_has_pathway[second] > clone_has_pathway[first]): return False

    return all(tree_has_pathway) if consistency_type == "strict" else any(tree_has_pathway)

def get_consistent_patients(CBNPM, patient_trees, consistency_type):
    # patient_trees is list of lists of trees
    # consistency_type is "strict" or "partial"
    
    consistent_patients = []
    
    for patient, trees in enumerate(patient_trees, start=1):
        if any(CBNPM_tumor_tree_consistency(CBNPM, tree, consistency_type) for tree in trees):
            consistent_patients.append(patient)

    return consistent_patients

def show_strictly_and_partially_consistent_patients(CBNPM, patient_trees):
    strictly_consistent_patients = get_consistent_patients(CBNPM, patient_trees, "strict")
    print(f"There are {len(strictly_consistent_patients)} strictly consistent patients: {strictly_consistent_patients}")

    partially_consistent_patients = get_consistent_patients(CBNPM, patient_trees, "partial")
    print(f"There are {len(partially_consistent_patients)} partially consistent patients: {partially_consistent_patients}")

def prune_CBNPM(CBNPM, consistent_patients_trees):
    alterations = set(mutation for trees in consistent_patients_trees for tree in trees for mutation in get_mutations_from_tree(tree))
        
    def process_tuple(node_tuple):
        return tuple(alteration for alteration in node_tuple if alteration in alterations)
        
    return nx.relabel_nodes(CBNPM, process_tuple, copy=True)

def get_patients_with_enough_alterations_dependencies_depth(patient_trees, num_alterations, num_dependencies, depth):
    def get_num_alterations_from_tree(tree):
        return len(get_mutations_from_tree(tree))

    def get_num_dependencies_from_tree(tree):
        num_edges = sum(1 for edge in tree.edges() if edge[0] != " ")

        num_cluster_dependencies = sum(len(node) - 1 for node in tree.nodes() if isinstance(node, tuple))
        return num_edges + num_cluster_dependencies

    def get_depth_from_tree(tree):
        branches = tumor_tree_to_branch_list(tree)
        
        def get_depth_for_branch(branch): return sum(len(node) if isinstance(node, tuple) else 1 for node in branch)
            
        return max(get_depth_for_branch(branch) for branch in branches) - 1 # - 1 because the branches included the root

    def tree_has_enough_alterations_dependencies_depth(tree, num_alterations, num_dependencies, depth):
        tree_num_alterations = get_num_alterations_from_tree(tree)
        tree_num_dependencies = get_num_dependencies_from_tree(tree)
        tree_depth = get_depth_from_tree(tree)

        return tree_num_alterations >= num_alterations and tree_num_dependencies >= num_dependencies and tree_depth >= depth

    patients = []
    
    for patient, trees in enumerate(patient_trees, start=1):
        if (any(tree_has_enough_alterations_dependencies_depth(tree, num_alterations, num_dependencies, depth) for tree in trees)):
            patients.append(patient)

    return patients

def trees_lists_equals(trees1, trees2):
    """doesn't require it to be a tree really, but just makes sure nodes and edges of each one are equal, including order of the trees/graphs, but for any nodes that are tuples, order doesn't matter"""

    def sort_node(node):
        return tuple(sorted(node)) if isinstance(node, tuple) else node

    def tree_equals(tree1, tree2):
        nodes1 = set(sort_node(node) for node in tree1.nodes())
        nodes2 = set(sort_node(node) for node in tree2.nodes())
        edges1 = set((sort_node(parent), sort_node(child)) for parent, child in tree1.edges())
        edges2 = set((sort_node(parent), sort_node(child)) for parent, child in tree2.edges())
        return nodes1 == nodes2 and edges1 == edges2

    return len(trees1) == len(trees2) and all(tree_equals(tree1, tree2) for tree1, tree2 in zip(trees1, trees2))

def get_tree_stage(progression_model, tumor_tree):
    """returns the stage of the tumor tree based on the progression model if partially consistent, otherwise returns -1
    the stage is the latest stage of all pathways mutated in the tumor tree
    the progression_model is a DAG on pathways, doesn't have a root node, tumor_tree is a tree on alterations or sets of alterations, has root node " " """

    if (not CBNPM_tumor_tree_consistency(progression_model, tumor_tree, "partial")): return -1

    def get_tree_depth(tree):
        branches = tumor_tree_to_branch_list(tree)
        
        def get_depth_for_branch(branch): return sum(len(node) if isinstance(node, tuple) else 1 for node in branch)
            
        return max(get_depth_for_branch(branch) for branch in branches) - 1 # - 1 because the branches included the root

    S = set(alteration for pathway in progression_model.nodes() for alteration in pathway) | {" "}
    collapsed_tumor_tree = make_collapsed_tree1(tumor_tree, S)

    return get_tree_depth(collapsed_tumor_tree)

def get_patient_stage(progression_model, trees):
    """a list of trees for a single patient"""

    return max(get_tree_stage(progression_model, tree) for tree in trees)

def get_TRACERx_patient_trees_lists():
    """returns list of lists (one per patient) of tumor trees (networkx digraphs)"""
    
    file_path = os.path.join(get_file_path("project_root"), "data", "tracerx", "tracerx_lung_phylogenies.txt")

    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f]

    patient_trees_lists = [] 
    num_patients = int(lines[0].split()[0])
    
    idx = 1
    for _ in range(num_patients):
        
        trees = []
        num_trees = int(lines[idx].split()[0]); idx += 1
        
        for _ in range(num_trees):
            tree = nx.DiGraph()
            num_nodes = int(lines[idx].split()[0]); idx += 1
            nodes = {}
            
            for _ in range(num_nodes):
                node = lines[idx].split(); idx += 1
                if (node[1] == "root"): nodes[node[0]] = " "
                elif (len(node[1].split(',')) == 1): nodes[node[0]] = node[1].split(',')[0]
                else: nodes[node[0]] = tuple(node[1].split(','))

            num_edges = int(lines[idx].split()[0]); idx += 1
            for _ in range(num_edges):
                edge = lines[idx].split(); idx += 1
                tree.add_edge(nodes[edge[0]], nodes[edge[1]])

            trees.append(tree)

        patient_trees_lists.append(trees)
    
    return patient_trees_lists

def TRACERx_patient_number_to_name(number):
    """inputs number 1-99, ouputs patient name"""

    patients_numbers = [f"CRUK{i:04d}" for i in range(1, 101) if i != 53]
    patient_map = {i: patient_name for i, patient_name in enumerate(patients_numbers, start=1)}

    return patient_map[number]

# ----------------------------------------------------------------------------------------------------
# main problem (rn named 1.1) (these comments are outdated btw)

def problem_1_1(patient_trees, alterations, num_pathways, num_dependencies, num_levels, max_solutions=1, max_alterations_per_pathway=None):
    """
    Problem: Given a cohort of n patients and their tumor trees, a set of alterations A, and a parameter K, find a TPM that maximizes the 
    number of consistent patients (scored by maximizing strict consistency as primary objective and partial consistency as secondary objective)
        
    This problem reduces to finding an LPM
    
    Constructs a clone-alteration matrix, constructs an ILP, runs it, and returns the solution LPM along with other info
    
    Params:
        patient_trees (list of lists of DiGraph's (each node is str or [tuple of strs]): 
            A list of patients [p_1, ..., p_n]. Each patient is a list of NetworkX DiGraph objects,
            where each one is a tree, and each node in a tree is a single alteration (str) or a cluster (tuple of strs); (the root node is " ")
        alterations (set of strs): A set of alterations
        num_pathways (int): the number of pathways
    
    Returns:
        results (dict with str keys): the results (keys and values are detailed below)
        
            LPM (NetworkX DiGraph): the LPM, where each node is a tuple of strs and represents a pathway
            pathways (dict with int keys, list (of strs) values): the pathways. pathways[0] = [" "] (the root), pathways[i] is pathway i
            strictly_consistent_patients (list of ints): the patient numbers who are strictly consistent with the LPM
            partially_consistent_patients (list of ints): the patient numbers who are partially, but not strictly consistent with the LPM
            text_summary (str): a string containing brief info about the ILP, which patients were strictly and partially consistent, and which alterations are in the pathways
            time_took_ILP (float): the amount of time the ILP took to run
            time_took_all (float): the amount of time this function took to run
            ILP_log (str): the gurobi solver log for the ILP
            ILP_num_stuff (str): says how many variables, constraints, and non-zeros the ILP has
            
            validation_results (dict with str keys): the results of validating the ILP solution (keys and values are detailed below)
            
                correct (bool): whether or not the following three things are true
                    every patient in strictly_consistent_patients is strictly consistent with the LPM implied by the pathways
                    every patient in partially_consistent_patients is partially, but not strictly, consistent with the LPM
                    every patient that is in neither strictly_consistent_patients nor partially_consistent_patients is not partially consistent with the LPM
                    
                strictly_consistent_dict (dict with int keys and bool values): keys are patient number, values are whether they are strictly consistent
                partially_consistent_dict (dict with int keys and bool values): keys are patient number, values are whether they are partially but not strictly consistent
                not_consistent_dict (dict with int keys and bool values): keys are patient number, values are whether they are not partially consistent
    """

    start_all = time.perf_counter()

    results = {}

    M1, M2, M3 = make_three_matrices(patient_trees, alterations)

    model, s, r, q, a, z, y, yy, x, xx, others = problem_1_1_make_ilp(M1, M2, M3, alterations, num_pathways, num_dependencies, num_levels, max_solutions, max_alterations_per_pathway)

    if (max_solutions > 1):
        model.setParam('PoolSearchMode', 2) # 2 = find the n best solutions
        model.setParam('PoolSolutions', max_solutions) # store up to max_solutions solutions
        model.setParam('PoolGap', 0) # only keep solutions within 0% of optimal
    
    model.setParam('LogToConsole', 0)
    model.setParam('LogFile', "temp.log")
    start_ILP = time.perf_counter()
    
    model.optimize()
    results['time_took_ILP'] = time.perf_counter() - start_ILP

    with open("temp.log", 'r') as f:
        results['ILP_log'] = f.read()
        os.remove("temp.log")
    results['ILP_num_stuff'] = f"Variables: {model.NumVars}\nConstraints: {model.NumConstrs}\nNon-zeros: {model.NumNZs}"

    results.update(problem_1_1_get_solution(patient_trees, alterations, num_pathways, num_dependencies, num_levels, model, s, r, x, xx, max_solutions))

    results['ILP_vars'] = {'s': model.getAttr('X', s), 'r': model.getAttr('X', r), 'q': model.getAttr('X', q), 'a': model.getAttr('X', a), 'z': model.getAttr('X', z), 'y': model.getAttr('X', y), 'x': model.getAttr('X', x), 'others': others} # this has to go after problem_1_1_get_solution() bc if model is infeasible it'll be error, but we want problem_1_1_get_solution() to throw that exception

    results['validation_results_list'] = problem_1_1_validate_solution(patient_trees, results['CBNPM_list'], results['strictly_consistent_patients_list'], results['partially_consistent_patients_list'])

    results['time_took_all'] = time.perf_counter() - start_all

    return results

def problem_1_1_display_results(results, save_path=None):
    """
    Displays the results for a run of problem 1.1

    Params:
        results (dict with str keys): the results (keys and values used in this function are detailed below, but results may have other items not used here)
        
            LPM (NetworkX DiGraph): the LPM, where each node is a tuple of strs and represents a pathway
            text_summary (str): a string containing brief info about the ILP, which patients were strictly and partially consistent, and which alterations are in the pathways
            time_took_ILP (float): the amount of time the ILP took to run
            time_took_all (float): the amount of time problem_1_1() took to run
            
            validation_results (dict with str keys): the results of validating the ILP solution (keys and values are detailed below)
            
                correct (bool): whether or not the following three things are true
                    every patient in strictly_consistent_patients is strictly consistent with the LPM implied by the pathways
                    every patient in partially_consistent_patients is partially, but not strictly, consistent with the LPM
                    every patient that is in neither strictly_consistent_patients nor partially_consistent_patients is not partially consistent with the LPM
                    
                strictly_consistent_dict (dict with int keys and bool values): keys are patient number, values are whether they are strictly consistent
                partially_consistent_dict (dict with int keys and bool values): keys are patient number, values are whether they are partially but not strictly consistent
                not_consistent_dict (dict with int keys and bool values): keys are patient number, values are whether they are not partially consistent
    """

    print(f"Running the ILP took {results['time_took_ILP']} seconds")
    print(f"Running all of problem_1_1() took {results['time_took_all']} seconds\n")

    print(results['text_summary'])

    for i, CBNPM in enumerate(results['CBNPM_list'], start=1):
        if (len(results['CBNPM_list']) > 1): print(f"For solution {i}")
        print("The CBNPM after removing alterations not present in any of the consistent patients:")

        show_tree(CBNPM, save_path=save_path)
        if (save_path is not None):
            print(f"The CBNPM was saved to {os.path.basename(save_path)}\nCBNPM nodes: {sorted(CBNPM.nodes())}\nCBNPM edges: {sorted(CBNPM.edges())}")

    for i, validation_results in enumerate(results['validation_results_list'], start=1):
        if (len(results['validation_results_list']) > 1): print(f"Validation results for for solution {i}")
        problem_1_1_display_validation_result(**validation_results)

def problem_1_1_make_ilp(clone_alteration_matrix, tree_clone_matrix, patient_tree_matrix, alterations, num_pathways, num_dependencies, num_levels, max_solutions=1, max_alterations_per_pathway=None):
    """Constructs an ILP for problem 1.1

    Uses the notion of clone consistency for LPMs (mutual exclusivity and ordering), a tumor tree can only be consistent if all of its clones are

    s[j,k] = 1 if alteration j is assigned to pathway k
    r[k,kk] = 1 if edge between pathway k and kk (k is necessary for kk)
    q[k,l] = 1 if pathway k is at level l
    a[i,k] = 1 if clone i has at least 1 alteration in pathway k
    z[i] = 1 if clone i is consistent with the chosen LPM
    y[t] = 1 if tree t is strictly consistent with the chosen LPM 
    x[p] = 1 if patient p is strictly consistent with the chosen LPM 
    
    Params:
        clone_alteration_matrix (list of dicts (each with str keys and int (1 or 0) values)): which clones (1st index) have which alterations (2nd index)
        tree_clone_matrix (list of lists of ints (1 or 0)): which trees (rows) have which clones (cols)
        patient_tree_matrix (list of lists of ints (1 or 0)): which patients (rows) have which trees (cols)
        alterations (set of strs): the set of alterations
        num_pathways (int): the number of pathways that the LPM must have

    Returns:
        model (gurobipy.Model): the ILP model
        s, a, z, y, yy, x, xx (each is a gurobipy.tupledict): the variables from the ilp solver. returned individually as s, a, z, y, yy, x, xx
    """

    # shorter names, and making 1-indexed
    M1 = [None] + clone_alteration_matrix
    M2 = [None] + [[0] + row for row in tree_clone_matrix]
    M3 = [None] + [[0] + row for row in patient_tree_matrix]

    pathway_indexes = range(1, num_pathways + 1)
    level_indexes = range(1, num_levels + 1)
    clone_indexes = range(1, len(M1))
    tree_indexes = range(1, len(M2))
    patient_indexes = range(1, len(M3))

    model = gp.Model("problem_1_1")
    
    s = model.addVars(alterations, pathway_indexes, vtype=GRB.BINARY, name="s")
    model.addConstrs(s.sum(j,'*') <= 1 for j in alterations) # each assigned to no more than one pathway
    model.addConstrs(s.sum('*',k) >= 1 for k in pathway_indexes) # no empty pathways
    if (max_alterations_per_pathway is not None): model.addConstrs(s.sum('*',k) <= max_alterations_per_pathway for k in pathway_indexes) # each pathway can't have more than the max number of alterations per pathway

    r = model.addVars(list(itertools.permutations(pathway_indexes, 2)), vtype=GRB.BINARY, name="r")
    model.addConstr(r.sum() == num_dependencies) # the only constraint needed is # dependencies. it doesn't need to be connected, and making it acyclic should happen automatically
    
    q = model.addVars(pathway_indexes, level_indexes, vtype=GRB.BINARY, name="q")
    model.addConstrs(q.sum(k,'*') == 1 for k in pathway_indexes) # each pathway belongs to exactly one level
    model.addConstrs(q.sum('*',l) >= 1 for l in level_indexes) # each level has at least one pathway
    model.addConstrs(q[k,1] >= 1 - r.sum('*',k) for k in pathway_indexes) # if no incoming edges, it is at level 1
    model.addConstrs(q[kk,l] >= r[k,kk] + q[k,l-1] - 1 for l in level_indexes[1:] for k, kk in itertools.permutations(pathway_indexes, 2)) # edges are only allowed between consecutive levels
    model.addConstrs(r.sum(k,'*') <= num_pathways * (1 - q[k,num_levels]) for k in pathway_indexes) # no edges allowed from nodes at last level (restricts more levels, and also fixes problem where the other constraints wouldn't apply if added more levels)

    a = model.addVars(clone_indexes, pathway_indexes, vtype=GRB.BINARY, name="a") # the following constraints properly set the value of the 'a' vars
    model.addConstrs(a[i,k] >= s[j,k] for j in alterations for i in clone_indexes for k in pathway_indexes if M1[i][j])
    model.addConstrs(a[i,k] <= gp.quicksum(s[j,k] for j in alterations if M1[i][j]) for i in clone_indexes for k in pathway_indexes)

    z = model.addVars(clone_indexes, vtype=GRB.BINARY, name="z") # clone consistency

    if (max_solutions == 1):
        model.addConstrs(z[i] <= 2 - r[k,kk] - a[i,kk] + a[i,k] for i in clone_indexes for k, kk in itertools.permutations(pathway_indexes, 2)) # basically if r[k,kk] = 1 and the clone has alteration in pathway kk but not k then it's not consistent
        model.addConstrs(gp.quicksum(s[j,k] for j in alterations if M1[i][j]) <= 1 + sum(M1[i][j] for j in alterations) * (1 - z[i]) for i in clone_indexes for k in pathway_indexes) # if clone has multiple alterations in same pathway, it's not consistent
        others = {}

    else:
        # a few helper vars to help remove slack for the case when we want to allow multiple equal-in-score but non-identical solutions
        zz = model.addVars(clone_indexes, vtype=GRB.BINARY, name="zz") # for whether each clone satisfies ordering (no slack)
        
        zz_helper = model.addVars(clone_indexes, list(itertools.permutations(pathway_indexes, 2)), vtype=GRB.BINARY, name="zz_helper") # whether or not there is a conflict for clone i and pathways k, kk (i.e., there's dependency (k, kk) but clone i has kk but not k)
        model.addConstrs(zz_helper[i,k,kk] >= r[k,kk] + a[i,kk] - a[i,k] - 1 for i in clone_indexes for k, kk in itertools.permutations(pathway_indexes, 2)) # if r[k,kk] = 1 and the clone has alteration in pathway kk but not k then there's a conflict
        model.addConstrs(zz_helper[i,k,kk] <= r[k,kk] for i in clone_indexes for k, kk in itertools.permutations(pathway_indexes, 2)) # if no dependency, there's no conflict
        model.addConstrs(zz_helper[i,k,kk] <= a[i,kk] for i in clone_indexes for k, kk in itertools.permutations(pathway_indexes, 2)) # if doesn't have pathway kk, there's no conflict
        model.addConstrs(zz_helper[i,k,kk] <= 1 - a[i,k] for i in clone_indexes for k, kk in itertools.permutations(pathway_indexes, 2)) # if has pathway k, there's no conflict
    
        model.addConstrs(zz[i] <= 1 - zz_helper[i,k,kk] for i in clone_indexes for k, kk in itertools.permutations(pathway_indexes, 2)) # if there's a conflict for any pathway pair, then the clone doesn't satisfy ordering
        model.addConstrs(zz[i] >= 1 - zz_helper.sum(i,'*','*') for i in clone_indexes) # if there's no conflict for all pathway pairs, then the clone satisfies ordering
    
        zzz = model.addVars(clone_indexes, vtype=GRB.BINARY, name="zzz") # for whether each clone satisfies mutual exclusivity (no slack) 
    
        zzz_helper = model.addVars(clone_indexes, pathway_indexes, vtype=GRB.BINARY, name="zzz_helper") # whether or not there is a conflict for clone i and pathway k (i.e., clone i has multiple alterations in pathway k
        model.addConstrs(zzz_helper[i,k] <= gp.quicksum(s[j,k] for j in alterations if M1[i][j]) / 2 for i in clone_indexes for k in pathway_indexes) # if there's less than two alterations in this clone and pathway, there's no conflict
        model.addConstrs(gp.quicksum(s[j,k] for j in alterations if M1[i][j]) <= 1 + sum(M1[i][j] for j in alterations) * zzz_helper[i,k] for i in clone_indexes for k in pathway_indexes) # if there's two or more alterations in this clone and pathway, then there's a conflict
    
        model.addConstrs(zzz[i] <= 1 - zzz_helper[i,k] for i in clone_indexes for k in pathway_indexes) # if there's a conflict for any pathway, then the clone doesn't satisfy mutual exclusivity
        model.addConstrs(zzz[i] >= 1 - zzz_helper.sum(i,'*') for i in clone_indexes) # if there's no conflict for all pathways, then the clone satisfies mutual exclusivity
    
        model.addConstrs(z[i] >= zz[i] + zzz[i] - 1 for i in clone_indexes) # if clone satisfies both ordering and mutual exclusivity, it's consistent
        model.addConstrs(z[i] <= zz[i] for i in clone_indexes) # if clone doesn't satisfy ordering, it's not consistent
        model.addConstrs(z[i] <= zzz[i] for i in clone_indexes) # if clone doesn't satisfy mutual exclusivity, it's not consistent

        others = {'zz': zz, 'zz_helper': zz_helper, 'zzz': zzz, 'zzz_helper': zzz_helper}

    y = model.addVars(tree_indexes, vtype=GRB.BINARY, name="y") # tree strict consistency
    model.addConstrs(y[t] <= z[i] for t in tree_indexes for i in clone_indexes if M2[t][i] == 1) # all tree's clones must be consistent for tree to be consistent
    model.addConstrs(y[t] <= gp.quicksum(a.sum(i,k) for i in clone_indexes if M2[t][i] == 1) for k in pathway_indexes for t in tree_indexes) # if there is any pathway which none of a tree's clones have, then the tree isn't striclty consistent

    yy = model.addVars(tree_indexes, vtype=GRB.BINARY, name="yy") # tree partial consistency
    model.addConstrs(yy[t] <= z[i] for t in tree_indexes for i in clone_indexes if M2[t][i] == 1)
    model.addConstrs(yy[t] <= gp.quicksum(a.sum(i,'*') for i in clone_indexes if M2[t][i] == 1) for t in tree_indexes) # for tree to be partially consistent, at least one clone must contain at least one pathway

    x = model.addVars(patient_indexes, vtype=GRB.BINARY, name="x") # patient strict consistency
    model.addConstrs(x[p] <= gp.quicksum(y[t] for t in tree_indexes if M3[p][t] == 1) for p in patient_indexes) # for a patient to be consistent, it must have at least one tree that is consistent

    xx = model.addVars(patient_indexes, vtype=GRB.BINARY, name="xx") # patient partial consistency
    model.addConstrs(xx[p] <= gp.quicksum(yy[t] for t in tree_indexes if M3[p][t] == 1) for p in patient_indexes)

    model.setObjective(x.sum() + xx.sum() / (len(patient_indexes) + 1), GRB.MAXIMIZE) # maximize strict consistency as primary objective, partial as secondary

    return model, s, r, q, a, z, y, yy, x, xx, others

def problem_1_1_get_solution(patient_trees, alterations, num_pathways, num_dependencies, num_levels, model, s, r, x, xx, max_solutions=1):
    """
    After running model.optimize() for the ILP, this function extracts the solution from it and returns info about the solution
    
    Params:
        patient_trees (list of lists of DiGraph's (each node is str or [tuple of strs]): 
            A list of patients [p_1, ..., p_n]. Each patient is a list of NetworkX DiGraph objects,
            where each one is a tree, and each node in a tree is a single alteration (str) or a cluster (tuple of strs); (the root node is " ")
        alterations (set of strs): A set of alterations
        num_pathways (int): the number of pathways
        model (gurobipy.Model): the ILP model
        s (gurobipy.tupledict): s[j,k] = 1 if alteration j is assigned to pathway k
        x (gurobipy.tupledict): x[p] = 1 if patient p is strictly consistent with the chosen LPM 
        xx (gurobipy.tupledict): xx[p] = 1 if patient p is partially consistent with the chosen LPM 

    Returns:
        results (dict with str keys): the results (keys and values are detailed below)
            LPM (NetworkX DiGraph): the LPM, where each node is a tuple of strs and represents a pathway
            pathways (dict with int keys, list (of strs) values): the pathways. pathways[0] = [" "] (the root), pathways[i] is pathway i
            strictly_consistent_patients (list of ints): the patient numbers who are strictly consistent with the LPM
            partially_consistent_patients (list of ints): the patient numbers who are partially, but not strictly consistent with the LPM
            text_summary (str): a string containing brief info about the ILP, which patients were strictly and partially consistent, and which alterations are in the pathways
    """ 

    output_buffer = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = output_buffer
    try:

        pathway_indexes = range(1, num_pathways + 1)
        patient_indexes = range(1, len(patient_trees) + 1)
    
        print(f"is MIP (should be 1): {model.IsMIP}") 
        print(f"is QCP (should be 0): {model.IsQCP}") 
        print(f"is optimal (should be True): {model.status == GRB.OPTIMAL}") 
    
        if (model.status != GRB.OPTIMAL): 
            raise Exception(f"model status: {model.status} (see documentation for what it means)")

        CBNPM_list = []
        pathways_list = []
        strictly_consistent_patients_list = []
        partially_consistent_patients_list = []

        if (max_solutions > 1):
            n_solutions = model.SolCount
            print(f"Number of solutions found: {n_solutions}")
            use_pool = True
        else: 
            n_solutions = 1 # apparently model.SolCount gives a different thing based on suboptimal solutions if model.setParam('PoolSearchMode', 2) wasn't set
            use_pool = False

        def get_val(var):
            return var.Xn if use_pool else var.X

        for i in range(n_solutions):
            if (n_solutions > 1): print(f"\n------------------------------SOLUTION {i+1}------------------------------\n")
            model.setParam('SolutionNumber', i)
        
            # print(f"Score: {model.PoolObjVal}\n") # this is same as objVal unless you allow multiple solutions within non-0% of optimal
            print(f"Score: {model.objVal}\n")

            strictly_consistent_patients = [p for p in patient_indexes if get_val(x[p]) > 0.5]
            num_eligible_patients = len(get_patients_with_enough_alterations_dependencies_depth(patient_trees, num_pathways, num_dependencies, num_levels))
            print(f"# strictly consistent patients (out of {num_eligible_patients} which had enough pathways, dependencies, and levels): {len(strictly_consistent_patients)}")
            print(f"Strictly consistent patients: {strictly_consistent_patients}\n")

            not_strictly_consistent_patients = set(patient_indexes) - set(strictly_consistent_patients)
    
            partially_consistent_patients = [p for p in not_strictly_consistent_patients if get_val(xx[p]) >= 0.5]
            print(f"# partially consistent patients (out of the remaining {len(patient_trees)} - {len(strictly_consistent_patients)} = {len(not_strictly_consistent_patients)} that weren't strictly consistent): {len(partially_consistent_patients)}")
            print(f"Partially consistent patients: {partially_consistent_patients}\n")

            print("Note that the following is from the ILP and is before the CBNPM was pruned to remove alterations not present in any of the consistent patients")

            alterations_used = [alteration for alteration in alterations if sum(get_val(s[alteration, k]) for k in pathway_indexes) > 0.5]
            print(f"alterations used (out of possible {len(alterations)}): {len(alterations_used)}")
            print(f"alterations used: {alterations_used}")
        
            pathways = {k: [alteration for alteration in alterations if get_val(s[alteration,k]) > 0.5] for k in pathway_indexes}
            print(f"Pathways: {pathways}")

            edges = []
            for (k, kk) in itertools.permutations(pathway_indexes, 2):
                if (r[k,kk].Xn > 0.5):
                    edges.append((k, kk))

            CBNPM = make_CBNPM(pathways, edges)

            pruned_CBNPM = prune_CBNPM(CBNPM, [patient_trees[patient-1] for patient in strictly_consistent_patients + partially_consistent_patients])

            CBNPM_list.append(pruned_CBNPM)
            pathways_list.append(pathways)
            strictly_consistent_patients_list.append(strictly_consistent_patients)
            partially_consistent_patients_list.append(partially_consistent_patients)
    
        return {'CBNPM_list': CBNPM_list, 'pathways_list': pathways_list, 'strictly_consistent_patients_list': strictly_consistent_patients_list, 'partially_consistent_patients_list': partially_consistent_patients_list, 'text_summary': output_buffer.getvalue()}

    finally:
        sys.stdout = original_stdout

def problem_1_1_validate_solution(patient_trees, CBNPM_list, strictly_consistent_patients_list, partially_consistent_patients_list):
    """
    After running model.optimize() for the ILP and extracting the solution, this function validates the solution by making sure that the patients it claimed to be
    strictly/partially consistent were in fact strictly/partially consistent, and the ones that it claimed weren't, weren't

    Params:
        patient_trees (list of lists of DiGraph's (each node is str or [tuple of strs]): 
            A list of patients [p_1, ..., p_n]. Each patient is a list of NetworkX DiGraph objects,
            where each one is a tree, and each node in a tree is a single alteration (str) or a cluster (tuple of strs); (the root node is " ")
        pathways (dict with int keys, list (of strs) values): the pathways. pathways[0] = [" "] (the root), pathways[i] is pathway i
        strictly_consistent_patients (list of ints): the patient numbers who are strictly consistent with the LPM
        partially_consistent_patients (list of ints): the patient numbers who are partially, but not strictly consistent with the LPM

    Returns:
        results (dict with str keys): the results (keys and values are detailed below)
            correct (bool): whether or not the following three things are true
                every patient in strictly_consistent_patients is strictly consistent with the LPM implied by the pathways
                every patient in partially_consistent_patients is partially, but not strictly, consistent with the LPM
                every patient that is in neither strictly_consistent_patients nor partially_consistent_patients is not partially consistent with the LPM
            strictly_consistent_dict (dict with int keys and bool values): keys are patient number, values are whether they are strictly consistent
            partially_consistent_dict (dict with int keys and bool values): keys are patient number, values are whether they are partially but not strictly consistent
            not_consistent_dict (dict with int keys and bool values): keys are patient number, values are whether they are not partially consistent
    """

    validation_results_list = []

    for CBNPM, strictly_consistent_patients, partially_consistent_patients in zip(CBNPM_list, strictly_consistent_patients_list, partially_consistent_patients_list):

        strictly_consistent_dict = {} # patient number: whether they are strictly consistent
        partially_consistent_dict = {} # patient number: whether they are partially but not strictly consistent
        not_consistent_dict = {} # patient number: whether they are not partially consistent
                
        for patient_number, trees in enumerate(patient_trees, start=1):
    
            CBNPM_tree_strict = any(CBNPM_tumor_tree_consistency(CBNPM, tree, "strict") for tree in trees)
            CBNPM_tree_partial = any(CBNPM_tumor_tree_consistency(CBNPM, tree, "partial") for tree in trees)
    
            if (patient_number in strictly_consistent_patients):
                strictly_consistent_dict[patient_number] = CBNPM_tree_strict
            elif (patient_number in partially_consistent_patients):
                partially_consistent_dict[patient_number] = CBNPM_tree_partial and not CBNPM_tree_strict
            else:
                not_consistent_dict[patient_number] = not (CBNPM_tree_strict or CBNPM_tree_partial)

        validation_results_list.append({'correct': all(list(strictly_consistent_dict.values()) + list(partially_consistent_dict.values()) + list(not_consistent_dict.values())), 'strictly_consistent_dict': strictly_consistent_dict, 'partially_consistent_dict': partially_consistent_dict, 'not_consistent_dict': not_consistent_dict})

    return validation_results_list

def problem_1_1_display_validation_result(correct, strictly_consistent_dict, partially_consistent_dict, not_consistent_dict):
    """
    Prints the result of problem_1_1_validate_solution()

    Params:
        correct (bool): whether or not the following three things are true
            every patient in strictly_consistent_patients is strictly consistent with the LPM implied by the pathways
            every patient in partially_consistent_patients is partially, but not strictly, consistent with the LPM
            every patient that is in neither strictly_consistent_patients nor partially_consistent_patients is not partially consistent with the LPM
        strictly_consistent_dict (dict with int keys and bool values): keys are patient number, values are whether they are strictly consistent
        partially_consistent_dict (dict with int keys and bool values): keys are patient number, values are whether they are partially but not strictly consistent
        not_consistent_dict (dict with int keys and bool values): keys are patient number, values are whether they are not partially consistent        
    """

    print(f"correct (as in, every strictly consistent patient according to ilp is in fact strictly consistent, every partially consistent patient is in fact partially but not strictly consistent, and and every patient not consistent is in fact not consistent):\n{correct}\n")

    if not correct:
        print(f"strictly_consistent_dict: {strictly_consistent_dict}\n")
        print(f"partially_consistent_dict: {partially_consistent_dict}\n")
        print(f"not_consistent_dict: {not_consistent_dict}\n")

        strictly_consistent_wrong = [patient for patient in strictly_consistent_dict.keys() if not strictly_consistent_dict[patient]]
        partially_consistent_wrong = [patient for patient in partially_consistent_dict.keys() if not partially_consistent_dict[patient]]
        not_consistent_wrong = [patient for patient in not_consistent_dict.keys() if not not_consistent_dict[patient]]

        if (len(strictly_consistent_wrong) + len(partially_consistent_wrong) + len(not_consistent_wrong) <= 10):
            for patient in strictly_consistent_wrong:
                print(f"ilp says patient {patient} strictly consistent, but CBNPM_tumor_tree_consistency says this is wrong")
            for patient in partially_consistent_wrong:
                print(f"ilp says patient {patient} partially (but not strictly) consistent, but CBNPM_tumor_tree_consistency says this is wrong")
            for patient in not_consistent_wrong:
                print(f"ilp says patient {patient} not consistent, but CBNPM_tumor_tree_consistency says this is wrong")

        else:
            print("more than 10 were wrong, see dicts")

def run_problem_1_1_and_save_result(patient_trees, min_alteration_frequency, num_stages, num_pathways, num_dependencies, max_alterations_per_pathway, num_null_datasets, seed, save_folder_path, TRACERx=False):
    """saves the typical output of problem_1_1 to save_folder_path, as well as the following
    returns list with [num_stages, min_alteration_frequency, num_alt, max_alterations_per_pathway, seed, num_strictly_consistent_patients, num_partially_consistent_patients, time_took_problem, time_took_other, avg_alterations_per_pathway, num_null_datasets, "p1_value", "effect_size"]
    (all significance testing uses the permute_patient_trees null distribution function)"""

    start = time.perf_counter()

    # setup

    os.makedirs(save_folder_path, exist_ok=True)
    log_path = os.path.join(save_folder_path, "log.txt")

    old_stdout = sys.stdout
    sys.stdout = open(log_path, "w")
    try:

        PM_path = os.path.join(save_folder_path, "PM.png")
        patient_stages_path = os.path.join(save_folder_path, "patient_stages.csv")
    
        # run ILP

        alterations = get_alterations_with_min_frequency(patient_trees, min_alteration_frequency)

        ilp_result = problem_1_1(patient_trees, alterations, num_pathways, num_dependencies, num_stages, max_solutions=1, max_alterations_per_pathway=max_alterations_per_pathway)

        print()
        problem_1_1_display_results(ilp_result, save_path=PM_path)

        PM = ilp_result['CBNPM_list'][0]
        
        patient_stages = {TRACERx_patient_number_to_name(number) if TRACERx else number: (stage if (stage := get_patient_stage(PM, patient)) != -1 else "NA") for number, patient, in enumerate(patient_trees, start=1)}
        pd.DataFrame(patient_stages.items(), columns=["patient", "stage"]).to_csv(patient_stages_path, index=False)

        # extract things to return, and to add to log.txt

        num_strictly_consistent_patients = len(ilp_result['strictly_consistent_patients_list'][0])
        num_partially_consistent_patients = len(ilp_result['partially_consistent_patients_list'][0])
        p1_value, num_null_as_good, real_score, null_scores = empirical_p1_value(get_stage_score, permute_patient_trees, PM, patient_trees, num_null_datasets, seed)
        effect_size = get_effect_size(real_score, null_scores)
        time_took_problem = ilp_result['time_took_all']
        avg_alterations_per_pathway = sum(len(node) if isinstance(node, tuple) else 1 for node in PM.nodes()) / len(PM.nodes()) # actually i think it is always tuple but just in case

        print(f"\n\np_1-value: {p1_value}")
        print(f"\nnumber of null datasets with >= score for the PM as real dataset does: {num_null_as_good}")
        print(f"\nreal score: {real_score}")
        print(f"\naverage null score: {np.mean(null_scores)}")
        print(f"\neffect size: {effect_size}")
        
        print(f"\naverage alterations per pathway: {avg_alterations_per_pathway}")

        print(f"number of alterations with at least {min_alteration_frequency} frequency: {len(alterations)}")

        return [num_stages, min_alteration_frequency, len(alterations), str(max_alterations_per_pathway), seed, num_strictly_consistent_patients, num_partially_consistent_patients, time_took_problem, time.perf_counter() - start - time_took_problem, avg_alterations_per_pathway, num_null_datasets, p1_value, effect_size]

    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

def run_problem_1_1_full_significance_testing(patient_trees, min_alteration_frequency, num_stages, num_pathways, num_dependencies, max_alterations_per_pathway, num_null_datasets, num_null_runs, seed, save_folder_path, TRACERx=False):
    
    """runs full significance testing for problem_1_1 involving 3 different scoring types, p1-values, effect sizes, and p2-values
    what this function does is basically a superset of p2_value()
    (all significance testing uses the permute_patient_trees null distribution function)
    writes results to files in folders for real and null runs
    num_null_datasets is the number of datasets we use for calculating p1-values and effect sizes
    num_null_runs is the number of times the whole problem is actually run on null datasets and is used for calculating p2-values
    returns list with the following:
    [seed, num_null_datasets, num_null_runs, total_time_took, total_time_took_problems, total_time_took_other,
    num_stages, num_pathways, num_dependencies, max_alterations_per_pathway, min_alteration_frequency, num_alterations, 
    true_correct, null_correct_list, true_time_took_problem, null_time_took_problem_list, true_time_took_other, null_time_took_other_list, 
    true_avg_alterations_per_pathway, null_avg_alterations_per_pathway_list, 
    true_strict_score, null_strict_score_list, true_strict_p1_value, null_strict_p1_value_list, true_strict_effect_size, null_strict_effect_size_list, strict_p2_value, 
    true_partial_score, null_partial_score_list, true_partial_p1_value, null_partial_p1_value_list, true_partial_effect_size, null_partial_effect_size_list, partial_p2_value, 
    true_stage_score, null_stage_score_list, true_stage_score_p1_value, null_stage_score_p1_value_list, true_stage_score_effect_size, null_stage_score_effect_size_list, stage_score_p2_value]"""

    total_time_took_start = time.perf_counter()

    if (num_null_runs < 1): raise Exception("please have num_null_runs at least one")

    random.seed(seed)

    alterations = get_alterations_with_min_frequency(patient_trees, min_alteration_frequency)

    def one_run(patient_trees, save_folder_path):

        time_took_start = time.perf_counter()

        # setup
    
        os.makedirs(save_folder_path, exist_ok=True)
        log_path = os.path.join(save_folder_path, "log.txt")
    
        old_stdout = sys.stdout
        sys.stdout = open(log_path, "w")
        try:
    
            PM_path = os.path.join(save_folder_path, "PM.png")
            patient_stages_path = os.path.join(save_folder_path, "patient_stages.csv")
        
            # run ILP
        
            ilp_result = problem_1_1(patient_trees, alterations, num_pathways, num_dependencies, num_stages, max_solutions=1, max_alterations_per_pathway=max_alterations_per_pathway)
    
            print()
            problem_1_1_display_results(ilp_result, save_path=PM_path)
    
            PM = ilp_result['CBNPM_list'][0]
            
            patient_stages = {TRACERx_patient_number_to_name(number) if TRACERx else number: (stage if (stage := get_patient_stage(PM, patient)) != -1 else "NA") for number, patient, in enumerate(patient_trees, start=1)}
            pd.DataFrame(patient_stages.items(), columns=["patient", "stage"]).to_csv(patient_stages_path, index=False)
    
            # extract things to return, and to add to log.txt

            correct = ilp_result['validation_results_list'][0]['correct']
            time_took_problem = ilp_result['time_took_all']
            avg_alterations_per_pathway = sum(len(node) if isinstance(node, tuple) else 1 for node in PM.nodes()) / len(PM.nodes()) # actually i think it is always tuple but just in case
    
            num_strictly_consistent_patients = len(ilp_result['strictly_consistent_patients_list'][0])
            strict_p1_value, _, real_strict_score, null_strict_scores = empirical_p1_value(get_strict_score, permute_patient_trees, PM, patient_trees, num_null_datasets)
            assert real_strict_score == num_strictly_consistent_patients
            strict_effect_size = get_effect_size(real_strict_score, null_strict_scores)

            num_partially_consistent_patients = len(ilp_result['partially_consistent_patients_list'][0])
            partial_p1_value, _, real_partial_score, null_partial_scores = empirical_p1_value(get_partial_score, permute_patient_trees, PM, patient_trees, num_null_datasets)
            assert real_partial_score == num_partially_consistent_patients + num_strictly_consistent_patients # the partial score counts patients that are strictly consistent as also partially consistent
            partial_effect_size = get_effect_size(real_partial_score, null_partial_scores)

            stage_score = get_stage_score(PM, patient_trees)
            stage_score_p1_value, _, real_stage_score, null_stage_scores = empirical_p1_value(get_stage_score, permute_patient_trees, PM, patient_trees, num_null_datasets)
            assert real_stage_score == stage_score
            stage_score_effect_size = get_effect_size(real_stage_score, null_stage_scores)

            print(f"\n\nnumber of alterations with at least {min_alteration_frequency} frequency: {len(alterations)}")
            print(f"\naverage alterations per pathway: {avg_alterations_per_pathway}")

            print(f"\nnum_null_datasets: {num_null_datasets}")
            print(f"\nstrict_p1_value: {strict_p1_value}")
            print(f"\nstrict_effect_size: {strict_effect_size}")
            print(f"\npartial_p1_value: {partial_p1_value}")
            print(f"\npartial_effect_size: {partial_effect_size}")
            print(f"\nstage_score_p1_value: {stage_score_p1_value}")
            print(f"\nstage_score_effect_size: {stage_score_effect_size}")
            
            time_took_other = time.perf_counter() - time_took_start - time_took_problem

            return [correct, time_took_problem, time_took_other, avg_alterations_per_pathway, 
                    real_strict_score, strict_p1_value, strict_effect_size, 
                    real_partial_score, partial_p1_value, partial_effect_size, 
                    stage_score, stage_score_p1_value, stage_score_effect_size]
    
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

    true_patient_trees = patient_trees
    true_save_folder_path = os.path.join(save_folder_path, "true")

    true_correct, true_time_took_problem, true_time_took_other, true_avg_alterations_per_pathway, true_strict_score, true_strict_p1_value, true_strict_effect_size, true_partial_score, true_partial_p1_value, true_partial_effect_size, true_stage_score, true_stage_score_p1_value, true_stage_score_effect_size = one_run(true_patient_trees, true_save_folder_path)

    null_results = []
    for i in range(1, num_null_runs + 1):
        null_patient_trees = permute_patient_trees(true_patient_trees)
        null_save_folder_path = os.path.join(save_folder_path, "null", f"null_{i}")
        
        null_results.append(one_run(null_patient_trees, null_save_folder_path))

    null_correct_list, null_time_took_problem_list, null_time_took_other_list, null_avg_alterations_per_pathway_list, null_strict_score_list, null_strict_p1_value_list, null_strict_effect_size_list, null_partial_score_list, null_partial_p1_value_list, null_partial_effect_size_list, null_stage_score_list, null_stage_score_p1_value_list, null_stage_score_effect_size_list = (list(null_result) for null_result in zip(*null_results))

    total_time_took = time.perf_counter() - total_time_took_start
    total_time_took_problems = true_time_took_problem + sum(null_time_took_problem_list)
    total_time_took_other = true_time_took_other + sum(null_time_took_other_list)

    num_alterations = len(alterations)

    #strict_p2_value = sum(null_strict_p1_value <= true_strict_p1_value for null_strict_p1_value in null_strict_p1_value_list) / num_null_runs
    strict_p2_value = sum(null_strict_effect_size >= true_strict_effect_size for null_strict_effect_size in null_strict_effect_size_list) / num_null_runs
    #partial_p2_value = sum(null_partial_p1_value <= true_partial_p1_value for null_partial_p1_value in null_partial_p1_value_list) / num_null_runs
    partial_p2_value = sum(null_partial_effect_size >= true_partial_effect_size for null_partial_effect_size in null_partial_effect_size_list) / num_null_runs
    #stage_score_p2_value = sum(null_stage_score_p1_value <= true_stage_score_p1_value for null_stage_score_p1_value in null_stage_score_p1_value_list) / num_null_runs
    stage_score_p2_value = sum(null_stage_score_effect_size >= true_stage_score_effect_size for null_stage_score_effect_size in null_stage_score_effect_size_list) / num_null_runs

    return [seed, num_null_datasets, num_null_runs, total_time_took, total_time_took_problems, total_time_took_other,
            num_stages, num_pathways, num_dependencies, str(max_alterations_per_pathway), min_alteration_frequency, num_alterations, 
            true_correct, null_correct_list, true_time_took_problem, null_time_took_problem_list, true_time_took_other, null_time_took_other_list, 
            true_avg_alterations_per_pathway, null_avg_alterations_per_pathway_list, 
            true_strict_score, null_strict_score_list, true_strict_p1_value, null_strict_p1_value_list, true_strict_effect_size, null_strict_effect_size_list, strict_p2_value, 
            true_partial_score, null_partial_score_list, true_partial_p1_value, null_partial_p1_value_list, true_partial_effect_size, null_partial_effect_size_list, partial_p2_value, 
            true_stage_score, null_stage_score_list, true_stage_score_p1_value, null_stage_score_p1_value_list, true_stage_score_effect_size, null_stage_score_effect_size_list, stage_score_p2_value]
    
# ----------------------------------------------------------------------------------------------------
# from pm.py

def is_leaf(G, node):
    """returns whether the given node is a leaf in the given graph
    raises an exception if node not in G"""

    if (node not in G):
        raise Exception("node not in G")
        
    return G.out_degree(node) == 0

def tumor_tree_to_branch_list(T):
    """given tumor tree as nx graph, returns a list of lists of mutations, where each inner list is a branch"""
    
    tree = []
    
    for node in T:
        if (is_leaf(T, node)):
            branch = [node]
            parent = node
            while parent != " ":
                parent = list(T.predecessors(parent))[0] # since it's a tree, just one predecessor
                branch.insert(0, parent)
            tree.append(branch)

    return tree

def get_width_from_font_size(font_size):
    """calculates a good width to use, from the given font size"""

    return int(150 / font_size)

def wrap_text(text, width1):
    """wraps the given text to have at most width characters in each line"""
    
    return textwrap.fill(text, width=width1)

def show_tree(T, my_font_size=7, save_path=None):    

    plt.figure(figsize=(8, 8), dpi=200)
    pos = nx.nx_pydot.graphviz_layout(T, prog='dot')

    node_labels = {node: str(node) for node in T.nodes()}
    wrapped_labels = {node: wrap_text(label, width1=get_width_from_font_size(my_font_size)) for node, label in node_labels.items()}

    nx.draw(T, pos, with_labels=False, node_color='lightgreen', node_size=4000, font_size=my_font_size)
    nx.draw_networkx_labels(T, pos, labels=wrapped_labels, font_size=my_font_size)

    edge_labels = nx.get_edge_attributes(T, 'weight')
    nx.draw_networkx_edge_labels(T, pos, edge_labels=edge_labels, font_size=my_font_size, font_color='red')

    if (save_path is not None):
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    else:      
        plt.show()
    plt.close()

def get_file_path(selection):
    """returns the file path of what is selected in selection, which must be "AML" for trees-aml.txt, "NSCLC" for trees-lung.txt, or "trees_visual" for the trees_visual dir, or "MASTRO_format_trees_files" for the MASTRO_format_trees_files dir or "project_root" for the project root"""

    this_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(this_file_path)))
    if (selection == "AML"): return os.path.join(project_root, "MASTRO_format_trees_files", "trees-aml.txt")
    elif (selection == "NSCLC"): return os.path.join(project_root, "MASTRO_format_trees_files", "trees-lung.txt")
    elif (selection == "trees_visual"): return os.path.join(project_root, "trees_visual")
    elif (selection == "MASTRO_format_trees_files"): return os.path.join(project_root, "MASTRO_format_trees_files")
    elif (selection == "project_root"): return project_root
    else: raise Exception("selection not recognized")

def get_all_alterations(dataset):
    """returns a list containing all alterations for the given dataset ("AML" or "NSCLC"), including the root ' ' """

    if (dataset == "AML"): return [' ', 'CBL', 'TP53', 'U2AF1', 'NRAS', 'ETV6', 'BCOR', 'SETBP1', 'KRAS', 'PPM1D', 'FLT3', 'DNMT3A', 'SMC3', 'CSF3R', 'IDH2', 'SRSF2', 'MYC', 'JAK2', 'RUNX1', 'PHF6', 'IDH1', 'NPM1', 'PTPN11', 'WT1', 'SF3B1', 'EZH2', 'GATA2', 'TET2', 'KIT', 'ASXL1', 'STAG2'] 
    elif (dataset == "NSCLC"): return [' ', 'CCND1amp', 'MET', 'KRAS', 'SOX2amp', 'NCOA6amp', 'RAD21', 'DICER1', 'COL2A1', 'PIK3CA', 'KEAP1', 'NFE2L2', 'NOTCH2', 'MAP3K1', 'FLT4', 'CMTR2', 'SGK223amp', 'CDKN2A', 'ARID1B', 'WT1', 'NCOR1', 'TP53', 'CREBBP', 'SERPINB13', 'NRAS', 'EP300', 'PASK', 'CBLB', 'FGFR1amp', 'DNM2', 'MGA', 'PRF1', 'FANCM', 'ATM', 'PTEN', 'EGFRamp', 'PTPRC', 'TERTamp', 'ARHGAP35', 'KRASamp', 'SPEN', 'PHOX2B', 'MYCamp', 'SETD2', 'NCOA6', 'FANCC', 'RB1', 'ARID2', 'SGK223', 'RNF43', 'MLH1', 'SMAD4', 'NF1', 'GATA3', 'U2AF1', 'APC', 'CTNNB1', 'WRN', 'CYLD', 'COL5A2amp', 'LATS1', 'KMT2D', 'CHEK2', 'METamp', 'BRAF', 'CIC', 'FAS', 'NFE2L2amp', 'STK11', 'IKZF1', 'PLXNB2', 'PRDM1', 'KMT2C', 'ASXL1', 'BAP1', 'PDGFRAamp', 'SMARCA4', 'FAT1', 'FBXW7', 'TSC2', 'RASA1', 'UBR5', 'EGFR', 'COL5A2', 'NOTCH1', 'CUX1', 'PIK3CAamp', 'POLE']
    else: raise Exception("selection not recognized")

def get_relationships_matrix(path_to_formatted_trees):
    """returns the relationships matrix for the given dataset ("AML" or "NSCLC")
    the relationships matrix is a list of lists (one per patient) of relationships, where a relationship is a string that looks like GENE-.-GENE where '.' can be '>', '/', or '?' """
    
    file = open(path_to_formatted_trees, "r")
    trees = file.readlines()
    return [tree.split() for tree in trees]

def relationship_type(relationship):
    """input is in form GENE-.-GENE where '.' can be '>', '/', or '?' """
    
    if ">" in relationship:
        return "ancestral"
    elif "/" in relationship:
        return "incomparable"
    elif "?" in relationship:
        return "clustered"
    else:
        print("something went wrong")
        exit()

# ----------------------------------------------------------------------------------------------------
# from pm2.py

def make_tree1(patient, dataset):
    """make_tree but can choose dataset rather than fixed to AML
    patient is a number (from 1, to the number of patients in the dataset)
    dataset is "AML" or "NSCLC" """

    if (dataset == "AML"): relationships = get_relationships_matrix(get_file_path("AML"))[patient - 1]
    elif (dataset == "NSCLC"): relationships = get_relationships_matrix(get_file_path("NSCLC"))[patient - 1]
    else: raise Exception("dataset not found")
    
    ancestral_relationships = [tuple(re.split('-.-', relationship)) for relationship in relationships if relationship_type(relationship) == "ancestral"]
    clustered_relationships = [tuple(re.split('-.-', relationship)) for relationship in relationships if relationship_type(relationship) == "clustered"]

    G = nx.DiGraph()
    
    for relationship in ancestral_relationships:
        G.add_edge(relationship[0], relationship[1])

    G = nx.transitive_reduction(G)

    genes_set = set(gene for pair in relationships for gene in tuple(re.split('-.-', pair)))
    for gene in genes_set:
        if (not G.has_node(gene) or G.in_degree(gene) == 0):
            G.add_edge(" ", gene)

    # new part
    clusters = [] # list of lists
    for (one, two) in clustered_relationships:
        
        one_index, two_index = -1, -1 # will be -1 if in no cluster
        for i, cluster in enumerate(clusters):
            if one in cluster:
                one_index = i
            if two in cluster:
                two_index = i

        if (one_index == -1 and two_index == -1):
            clusters.append([one, two])
        elif (one_index != -1 and two_index == -1):
            clusters[one_index].append(two)
        elif (two_index != -1 and one_index == -1):
            clusters[two_index].append(one)
        elif (one_index != two_index): # they are both in clusters but in different, so they have to be merged (this won't happen w/ trees-aml data but could w/ other)
             clusters[one_index].extend(clusters[two_index])
             del clusters[two_index]

    for cluster in clusters:
        G.add_node(tuple(cluster)) 
        for old_node in cluster:
            G = nx.contracted_nodes(G, tuple(cluster), old_node)

    return G

def get_mutations_from_tree(tree):
    """this is needed instead of just doing tree.nodes(), for the case of clusters
    returns a set"""

    mutations = []
    for node in tree.nodes():
        mutations.extend(node if isinstance(node, tuple) else [node])

    return set(mutations)
    
def get_patient_trees_lists(dataset):
    """gets a list (for patients) of lists (for tumor trees) for the given dataset ("AML" and "NSCLC" just have one tree per patient)
    dataset is "AML" or "NSCLC" or "TRACERx" """

    if (dataset == "TRACERx"): return get_TRACERx_patient_trees_lists()
    
    if (dataset == "AML"): patient_range = range(1, 121)
    elif (dataset == "NSCLC"): patient_range = range(1, 90)
    else: raise Exception("dataset not found")

    return [ [make_tree1(patient, dataset)] for patient in patient_range]

def show_patient_trees(patient_trees_lists, save_path=None, TRACERx=False):
    """patient_trees_lists is a list of lists of trees (one list per patient)
    pass in a save_path if you want to save the figure, otherwise doesn't save it
    if for TRACERx, TRACERx=True"""

    flat_trees = [(patient_num, tree_num, tree) for patient_num, trees in enumerate(patient_trees_lists, start=1) for tree_num, tree in enumerate(trees, start=1)]
    
    rows = int(np.ceil(np.sqrt(len(flat_trees))))
    cols = int(np.ceil(len(flat_trees) / rows))

    fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5), dpi=200)
    axs = np.ravel(axs)

    for ax, (patient_num, tree_num, T) in zip(axs, flat_trees):
        pos = nx.nx_pydot.graphviz_layout(T, prog='dot')
    
        node_labels = {node: str(node) for node in T.nodes()}
        wrapped_labels = {node: wrap_text(label, width1=get_width_from_font_size(8)) for node, label in node_labels.items()}

        nx.draw(T, pos, ax=ax, with_labels=False, node_color='lightgreen', node_size=1500, font_size=8)
        text_items = nx.draw_networkx_labels(T, pos, labels=wrapped_labels, font_size=8, ax=ax)

        for node, text_obj in text_items.items():
            label = wrapped_labels[node]
            text_obj.set_fontsize(max(5, 10 - (len(label) // 5)))

        edge_labels = nx.get_edge_attributes(T, 'weight')
        nx.draw_networkx_edge_labels(T, pos, edge_labels=edge_labels, font_size=8, font_color='red', ax=ax)

        if (len(patient_trees_lists[patient_num-1]) == 1):
            ax.set_title(f"Patient {patient_num}", fontsize=10)
        else:
            ax.set_title(f"Patient {patient_num if not TRACERx else TRACERx_patient_number_to_name(patient_num)}, Tree {tree_num}", fontsize=10)
        
        ax.axis('off')

    for ax in axs[len(flat_trees):]:
        ax.remove()

    plt.tight_layout()

    if (save_path is not None):
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def make_collapsed_tree1(tree, S):
    """input tree may have clusters (preserves them if multiple in S). make sure S includes the root " " """

    def is_kept(node):
        if isinstance(node, tuple):
            return any(m in S for m in node)
        return node in S

    nodes_to_keep = [n for n in tree.nodes() if is_kept(n)]

    T = nx.transitive_closure(tree)
    T = nx.induced_subgraph(T, nodes_to_keep)
    T = nx.transitive_reduction(T)

    mapping = {}
    for node in T.nodes():
        if isinstance(node, tuple):
            kept_mutations = tuple(m for m in node if m in S)
            mapping[node] = kept_mutations[0] if len(kept_mutations) == 1 else kept_mutations
        else:
            mapping[node] = node

    return nx.relabel_nodes(T, mapping, copy=True)
    