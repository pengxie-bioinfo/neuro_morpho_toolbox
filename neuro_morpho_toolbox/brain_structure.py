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
        df.columns = ["Abbrevation"] + [i for i in range(1, df.shape[1])]
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
        level = pd.DataFrame({'level': level, 'Abbrevation': df.Abbrevation.tolist(), 'Description': description},
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
                    print(j, cur_region, cur_level)
                elif (not cur_region is None) & (cur_level < level.loc[df.index[j], 'level']):
                    df_fill.iloc[j, i] = cur_region

        self.input_file = input_file
        self.df = df_fill
        self.level = level
        self.selected_regions = self.df.index.tolist()
        return

    def get_all_child_id(self, structure_id):
        cur_lvl = self.level.loc[structure_id]
        tp = self.df[self.df[cur_lvl.level]==cur_lvl['Description']]
        return tp.index.tolist()

    def get_selected_regions(self, input_file):
        brain_levels = pd.read_excel(input_file,
                                     usecols=[1, 3, 2, 5], index_col=0,
                                     names=['', 'Description', 'Abbrevation', 'level']
                                     )
        self.selected_regions = brain_levels.index.tolist()
        return




