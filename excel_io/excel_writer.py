import pandas as pd


def write_layout_and_unassigned(
    run_dims, bin_coords, unassigned_parts, output_file, bin_to_part_qty,
)->None:
    """Write warehouse layout sheets and an unassigned‐parts summary to an Excel file.

    Parameters
    ----------
    run_dims : pd.DataFrame
        DataFrame with columns 'Run No', '#Bays', and '#Levels' describing each run.
    bin_coords : dict[int, (run, column, level)]
        Mapping from bin_id to a tuple of (run number, column index, level index).
    placed : list of (part, bin_id, qty)
        List of placements returned by the model.
    unassigned_parts : dict[part, qty]
        Parts that could not be assigned, with their leftover quantities.
    output_file : str
        Path where the Excel workbook will be written.

    """
    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
        workbook = writer.book

        for _, run_row in run_dims.iterrows():
            run_number = int(run_row["Run No"])
            num_bays   = int(run_row["#Bays"])
            num_levels = int(run_row["#Levels"])

            # build an empty grid
            df = pd.DataFrame(
                "",
                index=range(1, num_levels + 1),
                columns=range(1, num_bays + 1),
            )

            # fill in occupied bins
            for bin_id, (run, col, lvl) in bin_coords.items():
                if run != run_number:
                    continue
                if bin_id in bin_to_part_qty:
                    part, qty = bin_to_part_qty[bin_id]
                    cell = f"{bin_id} ->\n{part}\n× {qty}"
                else:
                    cell = ""
                df.at[lvl, col] = cell

            # flip so highest level appears at the top
            df = df.iloc[::-1]

            # multi-line headings
            df.columns = [f"Column\n{c}" for c in df.columns]
            df.index   = [f"Level\n{l}"  for l in df.index]
            df.columns.name = None
            df.index.name   = None

            sheet_name = f"Run {run_number}"
            df.to_excel(writer, sheet_name=sheet_name, index=True)

            # enable text wrapping
            ws = writer.sheets[sheet_name]
            wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
            total_cols = df.shape[1] + 1  # including the index column
            ws.set_column(0, 0, 12, wrap_fmt)
            ws.set_column(1, total_cols - 1, 18, wrap_fmt)

        # write unassigned parts summary, if any
        if unassigned_parts:
            summary_df = pd.DataFrame(
                list(unassigned_parts.items()),
                columns=["Part Number", "Unassigned Qty"],
            )
            summary_df.to_excel(writer, sheet_name="Unassigned Parts", index=False)

    print(f"Warehouse layout has been written to {output_file}")
