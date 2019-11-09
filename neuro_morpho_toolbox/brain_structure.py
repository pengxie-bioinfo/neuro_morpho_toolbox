import numpy as np
import pandas as pd

class brain_structure:
    def __init__(self, input_file):
        # Read table
        MAXSIZE = 100
        my_cols = [i for i in range(MAXSIZE)]
        df = pd.read_csv(input_file, names=my_cols, engine='python',
                         skiprows=[0],
                         index_col=[1],
                         skipinitialspace=True)
        df = df.drop([0, 3, 4, 5, 6, 7, 8], axis=1)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                if df.iloc[i, j] is None:
                    df.iloc[i, j] = np.nan
        df.columns = ["Abbreviation"] + [i for i in range(1, df.shape[1])]
        df_isnull = df.isnull()

        # Get levels of each row
        level = []
        description = []
        for i in range(len(df)):
            for j in range(1, df.shape[1]):
                if not df_isnull.iloc[i, j]:
                    level.append(j)
                    description.append(df.iloc[i, j])
                    break
        MAXLEVEL = np.max(level)
        level = pd.DataFrame({'level': level, 'Abbreviation': df.Abbreviation.tolist(), 'Description': description},
                             index=df.index)
        
        # Drop redundant columns
        df = df.iloc[:, :(MAXLEVEL + 2)]  # The last column will contain only NaN
        df_isnull = df.isnull()

        # Fill empty slots in the table
        df_fill = df.copy()
        for i in range(1, df.shape[1]):
            cur_region = None
            for j in range(df.shape[0]):
                if not df_isnull.iloc[j, i]:
                    cur_region = df.iloc[j, i]
                    cur_level = level.loc[df.index[j], 'level']
                    # print(j, cur_region, cur_level)
                elif (not cur_region is None) & (cur_level < level.loc[df.index[j], 'level']):
                    df_fill.iloc[j, i] = cur_region

        self.input_file = input_file
        self.df = df_fill
        self.level = level
        self.selected_regions = self.df.index.tolist()
        self.dict_to_selected = {}
        for cur_region in self.selected_regions:
            child_ids = self.get_all_child_id(cur_region)
            for i in child_ids:
                self.dict_to_selected[i] = cur_region
        return

    def get_all_child_id(self, structure_id):
        if type(structure_id) == str:
            structure_id = nmt.bs.name_to_id(structure_id)
        cur_lvl = self.level.loc[structure_id]
        tp = self.df[self.df[cur_lvl.level]==cur_lvl['Description']]
        return tp.index.tolist()

    def get_selected_regions(self, input_file):
        brain_levels = pd.read_excel(input_file,
                                     usecols=[1, 2, 3, 5], index_col=0,
                                     names=['', 'Description', 'Abbreviation', 'level']
                                     )
        self.selected_regions = brain_levels.index.tolist()
        self.dict_to_selected = {}
        for cur_region in self.selected_regions:
            child_ids = self.get_all_child_id(cur_region)
            for i in child_ids:
                self.dict_to_selected[i] = cur_region
        return

    def name_to_id(self, region_name):
        # region_name can be either Abbreviation (checked first) or description
        tp = self.level[self.level.Abbreviation == region_name]
        if len(tp) != 0:
            return tp.index[0]
        tp = self.level[self.level.Description == region_name]
        if len(tp) != 0:
            return tp.index[0]
        print("Cannot find any regions named %s." % region_name)
        return -1

    def id_to_name(self, region_ID):
        # region_name can be either Abbreviation (checked first) or description
        if region_ID in self.level.index.tolist():
            return self.level.loc[region_ID,'Abbreviation']
        else:
            print("Cannot find any regions with ID %s." % region_ID)
        




