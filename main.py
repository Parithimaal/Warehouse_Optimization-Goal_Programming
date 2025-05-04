# %% [markdown]
# #### Imports

# %% [markdown]
# ##### Packages

# %%
from collections import defaultdict

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from excel_io import write_layout_and_unassigned
from fit_calc import qty_base_layer_factory, qty_max_factory
from utils import dict_from_attribute
from warehouse_geometry import Bin, Column, Dimensions

# %% [markdown]
# #### Reading Data

# %%
parts_file = "./Parts and Layout.xlsx"
output_file = "./Output-Warehouse Layout.xlsx"

# %%
parts_data = pd.read_excel(parts_file, sheet_name="Parts", index_col="Part Number")
run_dims = pd.read_excel(parts_file, sheet_name="Warehouse Layout")
bin_alterations = pd.read_excel(parts_file, sheet_name="Bin Altering")

# %% [markdown]
# ##### Imputing data types

# %%
stackable_col_map = {"yes": True, "no": False}
parts_data["Stackable"] = parts_data["Stackable"].map(stackable_col_map).astype(bool)
parts_data["Handpicable"] = parts_data["Handpickable"].map(stackable_col_map).astype(bool)  # noqa: E501

parts_data = parts_data.loc[~parts_data.index.isna()]
run_dims = run_dims.astype({"#Bays": np.int64, "#Levels": np.int64, "#Bins": np.int64})

# %% [markdown]
# ##### Storing bin merging data

# %%
merge_operations = []
bin_merge_data = bin_alterations["Bin to Merge"].dropna().astype(str)

if not bin_merge_data.empty:
    for to_merge in bin_merge_data["Bin to Merge"]:
        bins_to_merge = to_merge
        bins_to_merge = bins_to_merge.split(",")
        bins_to_merge = sorted([int(strr.strip()) for strr in bins_to_merge])
        merge_operations.append(tuple(bins_to_merge))

# %% [markdown]
# ##### Storing bin width extension data

# %%
bin_width_extensions = {}
bin_extension_data = bin_alterations[["Extend bin", "Change Width to"]].dropna()
for _, row in bin_extension_data.iterrows():
    bins_str = row["Extend bin"]
    if ":" in bins_str:
        bin_range = bins_str.split(":")
        bin_range = sorted([int(strr.strip()) for strr in bin_range])
        bin_ids = list(range(bin_range[0], bin_range[1]+1))
    else:
        bin_range = bins_str.split(",")
        bin_ids = sorted([int(strr.strip()) for strr in bin_range])
    for bin_id in bin_ids:
        bin_width_extensions[bin_id] = row["Change Width to"]

# %% [markdown]
# ##### Setting Goal Weights and Constants

# %%
goal_weights_data = pd.read_excel(
    "./Parts and Layout.xlsx", sheet_name="Goal Weights", index_col="Symbol in Model",
).drop(columns="Description")

goal_weights = goal_weights_data.Weight.to_dict()
w1, w2 = int(goal_weights["w1"]), int(goal_weights["w2"])
w3, w4 = int(goal_weights["w3"]), int(goal_weights["w4"])

HP_MAX_HEIGHT = float(goal_weights["HP_MAX_HEIGHT"])
BIN_PENALTY = float(goal_weights["BIN_PENALTY"])
SOLVER_TIME_LIMIT = float(goal_weights["SOLVER_TIME_LIMIT"])

# %% [markdown]
# #### Generating Bin Coordinates

# %%
bins = Bin.create_bins(run_dims)
columns = Column.interpolate_column_distances(run_dims)

# %%
if merge_operations:
    for to_merge in merge_operations:
        for idx, _ in enumerate(to_merge):
            left_bin_idx = 0
            left_bin_id = to_merge[0]
            if idx!=left_bin_idx:
                right_bin_id = to_merge[idx]
                bins[left_bin_id].merge_with(bins[right_bin_id])

# %%
for bin_id, bin_obj in bins.items():
    bin_obj.calculate_elevation()
    if bin_id in bin_width_extensions:
        bin_obj.extend_width_to(bin_width_extensions[bin_id])

# %%
bin_coords = {
    bin_id: (bins[bin_id].run, bins[bin_id].column[1], bins[bin_id].level)
    for bin_id in bins
}

# %% [markdown]
# #### Creating Dictionaries for Pyomo parameters
# %% [markdown]
# ##### Bins

# %%
distance_to_hp_door_b_ = dict_from_attribute(bins, "hp_door_dist")
distance_to_fl_door_b_ = dict_from_attribute(bins, "fl_door_dist")
height_b_ = dict_from_attribute(bins, "height")
elevation_b_ = dict_from_attribute(bins, "elevation")

# %% [markdown]
# ##### Parts

# %%

n_parts_p_ = parts_data["Stock Level"].to_dict()
is_stackable_p_ = parts_data.Stackable.to_dict()
is_handpickable_p_ = parts_data.Handpicable.astype(int).to_dict()
station_p_ = parts_data.Group.to_dict()
picks_per_week_p_ = parts_data["Picks per week"].to_dict()
dimensions_p_ = {
    part_num: Dimensions(row.Length, row.Width, row.Height)
    for part_num, row in parts_data.iterrows()
}

weight_p_ = parts_data["Weight (kg)"].to_dict()

# %% [markdown]
# #### Building feasible (p,b) pairs and per-pair max quantity

# %%
qty_base_layer = qty_base_layer_factory(dimensions = dimensions_p_, bins=bins)
qty_max = qty_max_factory(
    dimensions = dimensions_p_, bins=bins,
    is_stackable_p_=is_stackable_p_, qty_base_layer=qty_base_layer,
)

PB_feasible = [
    (p,b) for p in parts_data.index for b in bins
    if qty_max(p, b) > 0
]

capacities = {
    (p, b): qty_max(p, b) for (p, b) in PB_feasible
}

# %%
feasible_parts_set = {t[0] for t in PB_feasible}
len(feasible_parts_set)

oversized_parts = [part for part in n_parts_p_ if part not in feasible_parts_set]

if len(oversized_parts) > 0:
    error_msg = f"""\
    The following parts are too large to fit into any bin: {oversized_parts}."""
    "Do you want to resize or merge bins?"

    raise RuntimeError(error_msg)

# %% [markdown]
# #### Building Pyomo Model

# %% [markdown]
# ##### Setting Indices

# %%
m = pyo.ConcreteModel()
m.B = pyo.Set(initialize=bins.keys())
m.P = pyo.Set(initialize=parts_data.index)
m.PB = pyo.Set(dimen=2, initialize=PB_feasible)

# %% [markdown]
# ##### Parameters

# %% [markdown]
# ###### Bin Parameters

# %%
m.distance_to_hp_door_b = pyo.Param(m.B, initialize=distance_to_hp_door_b_)
m.distance_to_fl_door_b = pyo.Param(m.B, initialize=distance_to_fl_door_b_)
m.height_b = pyo.Param(m.B, initialize=height_b_)
m.elevation_b = pyo.Param(m.B, initialize=elevation_b_)


# %% [markdown]
# ###### Part Parameters

# %%
m.is_stackable_p = pyo.Param(m.P, initialize=is_stackable_p_, within=pyo.Binary)
m.is_handpickable_p = pyo.Param(m.P, initialize=is_handpickable_p_, within=pyo.Binary)
m.picks_per_week_p = pyo.Param(m.P, initialize=picks_per_week_p_)
m.n_parts_p = pyo.Param(m.P, initialize=n_parts_p_)
m.weight_p = pyo.Param(m.P, initialize=weight_p_)

# %% [markdown]
# ###### Part-Bin Parameters

# %%
m.capacity_p_b = pyo.Param(m.PB, initialize=capacities)

# %% [markdown]
# ##### Decision Variables

# %%
m.num_p_in_b = pyo.Var(m.P, m.B, within=pyo.NonNegativeIntegers)
m.is_part_in_bin = pyo.Var(m.P, m.B, within=pyo.Binary)

# %% [markdown]
# ##### Hard Constraints

# %% [markdown]
# ###### One SKU per bin Constraint

# %%
def one_sku_rule(m, b):  # noqa: ANN201
    return sum(m.is_part_in_bin[p, b] for p in m.P if (p, b) in m.PB)  <= 1
m.one_sku = pyo.Constraint(m.B, rule=one_sku_rule)

# %% [markdown]
# ###### Linking constraint - capacity, num_p_in_b and is_part_in_bin

# %%
def linking_rule(m, p, b):  # noqa: ANN201
    return m.num_p_in_b[p, b] <= m.capacity_p_b[p, b] * m.is_part_in_bin[p, b]
m.linking = pyo.Constraint(m.PB, rule=linking_rule)

# %% [markdown]
# ###### Compulsary assignment constraint

# %%
def compul_assign_rule(m, p):  # noqa: ANN201
    return sum(m.num_p_in_b[p, b] for b in m.B if (p, b) in m.PB) == m.n_parts_p[p]
m.compul_assign = pyo.Constraint(m.P, rule=compul_assign_rule)

# %% [markdown]
# ##### Soft Constraints (Goals)

# %%
# Deviation variables for goals
m.d1p = pyo.Var(within=pyo.NonNegativeReals)
m.d2p = pyo.Var(within=pyo.NonNegativeReals)
m.d3p = pyo.Var(within=pyo.NonNegativeReals)
m.d4p = pyo.Var(within=pyo.NonNegativeReals)


# %% [markdown]
# ###### Minimizing Handpicking Travel Distance

# %%
hp_dist_expr = sum(
    m.picks_per_week_p[p] * m.distance_to_hp_door_b[b] * m.num_p_in_b[p, b]
    for (p,b) in m.PB if m.is_handpickable_p[p]
)
m.hp_dist_goal_min = pyo.Constraint(expr=hp_dist_expr - m.d1p == 0)

# %% [markdown]
# ###### Minimizing Forklift Travel Distance

# %%
fl_dist_expr = sum(
    m.picks_per_week_p[p] * m.distance_to_fl_door_b[b] * m.num_p_in_b[p, b]
    for (p,b) in m.PB if not m.is_handpickable_p[p]
)
m.fl_dist_goal_min = pyo.Constraint(expr=fl_dist_expr - m.d2p == 0)

# %% [markdown]
# ###### Keep Handpickables below safe reach

# %%
safe_reach_expr = sum(
    m.num_p_in_b[p, b]
    * max(0, m.elevation_b[b] - HP_MAX_HEIGHT)
    * m.is_handpickable_p[p]
    for (p, b) in m.PB
)
m.safe_reach_goal_min = pyo.Constraint(expr=safe_reach_expr - m.d3p == 0)

# %% [markdown]
# ###### Minimize the elevation of heavy weight items

# %%
heavy_weight_expr = sum(m.num_p_in_b[p, b] * m.elevation_b[b] * m.weight_p[p]
                        for (p,b) in m.PB)
m.heavy_weight_goal = pyo.Constraint(expr=heavy_weight_expr - m.d4p == 0)

# %% [markdown]
# ##### Objective

# %%
m.obj = pyo.Objective(
    expr = w1 * m.d1p
            + w2 * m.d2p
            + w3 * m.d3p
            + w4 * m.d4p
            + BIN_PENALTY * sum(m.is_part_in_bin[p, b] for (p,b) in m.PB),
            sense=pyo.minimize,
)

# %% [markdown]
# ##### Solve

# %%
solver = pyo.SolverFactory("cbc")
solver_options = {}
if SOLVER_TIME_LIMIT is not None:
    solver_options["sec"] = SOLVER_TIME_LIMIT
results = solver.solve(m, options=solver_options, tee=True)

# %%
placed = [
    (p, b, int(m.num_p_in_b[p, b]()))
    for (p, b) in m.PB if m.num_p_in_b[p, b]() > 0
]
placed_sorted_by_b = sorted(placed, key=lambda t: t[1])

total_count_p = defaultdict(int)
assigned_bins = []
for p, b, qty in placed_sorted_by_b:
    assigned_bins.append(b)
    total_count_p[p] += qty

# %%
skipped_bins = set(bins.keys()) - set(assigned_bins)
skipped_bins

# %%
unassigned_parts = {}
for part, count in n_parts_p_.items():
    curr_count = count
    if part in total_count_p:
        curr_count -= total_count_p[part]

    if count!=0:
        unassigned_parts[part] = curr_count

# %% [markdown]
# #### Output

# %%
bin_to_part_qty = {b: (p, q) for (p, b, q) in placed}

write_layout_and_unassigned(
    run_dims, bin_coords, unassigned_parts, output_file, bin_to_part_qty,
)


