import numpy as np
import pandas as pd
from .column import Column
from collections import namedtuple

Dimensions = namedtuple('Dimensions', ['length', 'width', 'height']) # Creating a namedtuple class

class Bin:
    bins_dict = {}

    def __init__(self, bin_id, run, bay, level, level_height, bin_length, bin_width, run_col_coord, num_bins):
        self.bin_id = bin_id 
        self.run = run
        self.bay = bay
        self.level = level
        self.num_bins = num_bins
        self.level_height = level_height 

        self.length = bin_length
        self.width = bin_width
        self.height = level_height
        self.elevation = None

        self.column = run_col_coord
        Bin.bins_dict[bin_id] = self 

    @classmethod
    def create_bins(cls, run_dims):
        next_bin_id = 1

        for _, row in run_dims.iterrows():
            run_no = int(row['Run No'])
            num_bays = int(row['#Bays'])
            num_levels = int(row['#Levels'])
            num_bins = int(row['#Bins'])
            level_height = row["Level Height"]
            length = row["Bin Length"]
            width = row["Run Width"]

            for level in range(1, num_levels + 1):
                column = 1 # Initializing column for each level
                for bay in range(1, num_bays + 1):
                    for _bin in range(1, num_bins + 1):
                        run_col_coord = (run_no, column)
                        cls(next_bin_id, run_no, bay, level, level_height, length, width, run_col_coord, num_bins)
                        column += 1
                        next_bin_id += 1  # Increment the id counter counter for the next bin
        return cls.bins_dict
    
    def calculate_elevation(self):
        self.elevation = self.level_height * (self.level-1)
    
    @property
    def fl_door_dist(self):
        column = Column.columns_dict.get(self.column)
        return column.fl_dist

    @property            
    def hp_door_dist(self):
        column = Column.columns_dict.get(self.column)
        return column.hp_dist
    

    def merge_with(self, other):
        same_run = self.run == other.run 
        same_level = self.level == other.level
        same_bay = self.bay == other.bay
        consecutive = abs(self.bin_id - other.bin_id) == 1

        if same_run and same_level and same_bay and consecutive:
            self.length += other.length
            del Bin.bins_dict[other.bin_id]
            del other
        
        else:
            raise RuntimeError("The bins must be consecutive for merging")
    
    def extend_width_to(self, new_width):
        self.width = new_width