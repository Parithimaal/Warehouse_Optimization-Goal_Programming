# Warehouse Optimization using Goal Programming

This project solves the **warehouse slotting problem** using a multi-objective **goal programming model** in Pyomo. It allocates parts to bins while minimizing:

- Handpicking travel distance
- Forklift travel distance
- Unsafe elevations for handpickables
- Elevation of heavy items  

subject to capacity, stackability, and item compatibility constraints.
Refer to the attached "Goal Programming for Warehouse Allocation.pdf" for information on the mathematical model and the literature review.

## Folder Structure
```
warehouse_optimization/
├── excel_io/             # Excel output writer 
├── fit_calc/             # Quantity calculators for bin fit
├── utils/                # Generic utility functions
├── warehouse_geometry/   # Classes for Bin and Column definitions
├── main.py               # Core pipeline: data processing, model building, optimization
├── Parts and Layout.xlsx # Input file (Parts, Layout, Goal Weights)
├── Output-Warehouse Layout.xlsx # Output with bin assignments
├── LICENSE
├── README.md
```

## Optimization Model
The Pyomo model includes:

- **Decision Variables**
  - `num_p_in_b[p,b]`: Quantity of part `p` in bin `b`
  - `is_part_in_bin[p,b]`: Binary indicator if part `p` is in bin `b`

- **Constraints**
  - One SKU per bin
  - Part assignment must match total stock
  - Quantity bounds based on stacking and base-layer fit

- **Goals (Soft Constraints)**
  - Minimize walking distances (handpick/forklift)
  - Place heavy parts low
  - Avoid placing handpickables above ergonomic height

- **Objective**
  - Weighted sum of all goals + penalty for bin usage

## Sample Output
Output Excel contains:
- One sheet per warehouse run showing the layout and part assignments.
- One sheet listing any unassigned parts.

Example cell content in layout:
```
34 ->
P107
× 3
```

## How to Run

### 1. Install Dependencies
```bash
conda create --name <your_env_name> --file requirements.txt
conda activate <your_env_name>
```
CBC solver installation required

### 2. Prepare Input File
Modify `Parts and Layout.xlsx` with:
- Sheet `Parts`: Part specs
- Sheet `Warehouse Layout`: Run and bin details
- Sheet `Bin Altering`: Bin merges and width extensions
- Sheet `Goal Weights`: Weights for optimization terms

### 3. Run Optimization
```bash
python main.py
```
Output will be written to `Output-Warehouse Layout.xlsx`.

## License

This project is licensed under the [MIT License](./LICENSE).

## Acknowledgements

This project was developed as part of a research study on warehouse allocation optimization, supported by Alstom’s supply chain team.
