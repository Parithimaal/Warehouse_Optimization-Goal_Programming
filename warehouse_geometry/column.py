import numpy as np
import pandas as pd

class Column:
    columns_dict = {}

    def __init__(self, column_id, fl_dist, hp_dist):
        self.column_id = column_id
        self.fl_dist = fl_dist
        self.hp_dist = hp_dist
        Column.columns_dict[column_id] = self
   
    @classmethod
    def interpolate_column_distances(cls, run_dims):
        for index, row in run_dims.iterrows():
            run_number = row['Run No']
            run_front_dist_fl = row['Front Dist to ent: FL']
            run_back_dist_fl = row['Back Dist to ent: FL']
            run_front_dist_hp = row['Front Dist to ent: HP']
            run_back_dist_hp = row['Back Dist to ent: HP']

            num_bins = int(row['#Bins'])
            num_bays = int(row['#Bays'])
            num_columns = num_bins*num_bays

            fl_dists = np.round(np.linspace(run_front_dist_fl, run_back_dist_fl, num_columns), 2)
            hp_dists = np.round(np.linspace(run_front_dist_hp, run_back_dist_hp, num_columns), 2)
            
            for column_number, (fl_dist, hp_dist)  in enumerate(zip(fl_dists, hp_dists), start=1):
                column_id = (int(run_number), column_number)
                cls(column_id, fl_dist, hp_dist) 
        return cls.columns_dict