import neuro_morpho_toolbox as nmt
from neuro_morpho_toolbox import neuron
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import scale

class features:

    def __init__(self, feature_set_name):
        assert type(feature_set_name) == str, "'feature_set_name' must be a python string"
        self.feature_set_name = feature_set_name
        self.raw_data = pd.DataFrame()
        self.metadata = pd.DataFrame()
        return

    def add_raw_data(self, raw_data):
        assert type(raw_data) == pd.core.frame.DataFrame, "raw_data must be a pandas dataframe [n_neurons, n_features]"
        print("Number of input neurons: %d" % raw_data.shape[0])
        print("Number of input features: %d" % raw_data.shape[1])
        self.raw_data = raw_data.copy()
        # Basic filter: remove samples with all zeros
        remove_cells = self.raw_data.index[np.sum(self.raw_data, axis=1)==0].tolist()
        if len(remove_cells) > 0:
            print("All values are zeros for the following cells:")
            for i in remove_cells:
                print(i)
            # self.raw_data = self.raw_data.drop(remove_cells, axis=0)
        return

    def neuron_list(self):
        return self.raw_data.index.tolist()

    def feature_list(self):
        return self.raw_data.columns.tolist()

##########################################################
# Projection features
##########################################################
def read_location_table(csv_path, input_table):
    # 1. read a table
    if not input_table.endswith(".csv"):
        return None
    input_table = os.path.join(csv_path, input_table)
    df = pd.read_csv(input_table)
    df.index = df.structure_id

    # 1.1 Fill missing columns with NaN
    standard_columns = ['Unnamed: 0', '(basal) dendrite', 'apical dendrite',
                        'axon', 'hemisphere_id', 'soma', 'structure_id']
    for col in standard_columns:
        if col not in df.columns.tolist():
            df[col] = np.NaN
    df = df.fillna(0)
    # # 1.2 Delete regions that are not in the 'brain_regions' list
    # lab = []
    # for i in range(len(df)):
    #     if not df.index[i] in brain_levels.index.tolist():
    #         lab.append(df.index[i])
    # df = df.drop(lab)
    return df

def get_hemisphere(df):
    hemi_df = df[df.soma > 0]['hemisphere_id']
    if (len(hemi_df) > 0):
        if (len(set(hemi_df.tolist())) > 1):
            print("soma found in both hemispheres")
            return np.NAN
        else:
            hemi_id = hemi_df.iloc[0]
            return hemi_id

    # Find hemisphere by dendrite.
    df["dendrite"] = np.sum(df[["(basal) dendrite", "apical dendrite"]], axis=1)
    if np.sum(df["dendrite"]) == 0:
        print("no soma found")
        hemi_id = np.NAN
    if np.sum(df[df["hemisphere_id"] == 1]["dendrite"]) > np.sum(df[df["hemisphere_id"] == 2]["dendrite"]):
        hemi_id = 1
    else:
        hemi_id = 2
    return hemi_id


def add_new_record(df_dict, new_df, hemi_id, cell_id):
    new_df = new_df[new_df.hemisphere_id == hemi_id]

    for neurite_type, neurite_df in df_dict.items():
        tp = pd.DataFrame(np.zeros((1, neurite_df.shape[1])), columns=neurite_df.columns, index=[cell_id])
        common_features = list(set(new_df.index.tolist()).intersection(neurite_df.columns.tolist()))
        tp[common_features] = new_df.loc[common_features, neurite_type].tolist()
        df_dict[neurite_type] = pd.concat([neurite_df, tp])
    return df_dict

def initiate_df_dict():
    result = {}
    for i in nmt.neurite_types:
        result[i] = pd.DataFrame(columns=nmt.bs.selected_regions)
    return result

class projection_features(features):
    def __init__(self):
        features.__init__(self, "Projection")
        self.scaled_data = pd.DataFrame()
        return

    def load_csv_from_path(self, path):
        self.csv_path = path
        self.metadata = pd.DataFrame(columns=['File_name', 'Hemisphere'])
        hemi_dict = {1: initiate_df_dict(), 2: initiate_df_dict()}
        for input_table in sorted(os.listdir(path)):
            df = read_location_table(path, input_table)
            if df is None:
                continue
            cell_name = input_table.replace(".csv", "").replace(".swc", "").replace(".eswc", "")

            # Metadata
            hemi_id = get_hemisphere(df)
            if np.isnan(hemi_id):
                print(input_table)
                continue
            cur_metadata = pd.DataFrame({'File_name':os.path.join(path, input_table),
                                         'Hemisphere':hemi_id
                                         },
                                        index=[cell_name])
            self.metadata = pd.concat([self.metadata, cur_metadata])

            # Feature table
            for i in [1,2]:
                hemi_dict[i] = add_new_record(hemi_dict[i], df, i, cell_name)

        # ipsi_col = ["ipsi_" + nmt.bs.level.loc[i, "Abbrevation"] for i in hemi_dict[1]["soma"].columns.tolist()]
        ipsi_col = ["ipsi_" + nmt.bs.level.loc[i, "Abbrevation"] for i in hemi_dict[1]["axon"].columns.tolist()]
        contra_col = ["contra_" + nmt.bs.level.loc[i, "Abbrevation"] for i in hemi_dict[1]["axon"].columns.tolist()]
        axon_location = pd.DataFrame(index=hemi_dict[1]["axon"].index, columns=ipsi_col + contra_col, dtype='float32')
        # Flip the matrix if hemisphere == 2
        for i in range(len(self.metadata)):
            cell = self.metadata.index[i]
            hemi = self.metadata.Hemisphere[i]
            if hemi == 1:
                axon_location.loc[cell] = hemi_dict[1]["axon"].loc[cell].tolist() + \
                                          hemi_dict[2]["axon"].loc[cell].tolist()
            if hemi == 2:
                axon_location.loc[cell] = hemi_dict[2]["axon"].loc[cell].tolist() + \
                                          hemi_dict[1]["axon"].loc[cell].tolist()
        self.add_raw_data(axon_location)
        return

    def load_data_from_neuron_dict(self, neuron_dict):
        assert type(neuron_dict) == dict, "Error: projection_features.load_data_from_neuron_dict(self, neuron_dict).\nneuron_dict provided is NOT a python dictionary."
        for cur_name, cur_neuron in neuron_dict.items():
            assert type(cur_neuron) == neuron, "Error: projection_features.load_data_from_neuron_dict(self, neuron_dict).\nvalue of neuron_dict is NOT a nmt.neuron."
        region_used = nmt.bs.selected_regions
        columns = ["ipsi_"+nmt.bs.level.loc[i, "Abbrevation"] for i in region_used] + \
                  ["contra_"+nmt.bs.level.loc[i, "Abbrevation"] for i in region_used]

        df = pd.DataFrame(columns=columns)
        # TODO: show progress...
        for cur_name, cur_neuron in list(neuron_dict.items()):
            cur_df = cur_neuron.get_region_matrix(annotation=nmt.annotation,
                                                  brain_structure=nmt.bs,
                                                  region_used=None)
            # cur_df.index = cur_df.structure_id
            # print(cur_df.axon)
            if cur_neuron.hemi == 1:
                df.loc[cur_name] = cur_df.loc[cur_df["hemisphere_id"] == 1, "axon"].tolist() + \
                                   cur_df.loc[cur_df["hemisphere_id"] == 2, "axon"].tolist()
            else:
                df.loc[cur_name] = cur_df.loc[cur_df["hemisphere_id"] == 2, "axon"].tolist() + \
                                   cur_df.loc[cur_df["hemisphere_id"] == 1, "axon"].tolist()
        self.add_raw_data(df)
        self.normalize(log=True)
        return


    def normalize(self, log=True):
        scaled_data = np.array(self.raw_data) / np.sum(self.raw_data, axis=1).values.reshape(-1,1) * 100000
        if log:
            scaled_data = np.log(scaled_data+100)
        self.scaled_data = pd.DataFrame(scaled_data, index=self.raw_data.index, columns=self.raw_data.columns)
        return

class soma_features(features):
    def __init__(self):
        features.__init__(self, "Soma")
        self.raw_data = pd.DataFrame(columns=["x", "y", "z"])
        self.region = pd.DataFrame(columns=['Hemisphere', 'Region'])
        return

    def load_data_from_neuron_dict(self, neuron_dict):
        assert type(neuron_dict) == dict, "Error: soma_features.load_data_from_neuron_dict(self, neuron_dict).\n" \
                                          "neuron_dict provided is NOT a python dictionary."
        for cur_name, cur_neuron in neuron_dict.items():
            assert type(cur_neuron) == neuron, "Error: soma_features.load_data_from_neuron_dict(self, neuron_dict).\n" \
                                               "value of neuron_dict is NOT a nmt.neuron."

        for cur_name, cur_neuron in neuron_dict.items():
            df = cur_neuron.soma
            # XYZ
            self.raw_data = pd.concat([self.raw_data, df])

            # Region_table
            scale_data = [int(df["x"] / nmt.annotation.space["x"]),
                          int(df["y"] / nmt.annotation.space["y"]),
                          int(df["z"] / nmt.annotation.space["z"])
                          ]
            if df.z[0] < (nmt.annotation.micron_size['z']/2):
                cur_hemi = 1
            else:
                cur_hemi = 2
            structure_id = nmt.annotation.array[scale_data[0], scale_data[1], scale_data[2]]
            if structure_id in nmt.bs.dict_to_selected.keys():
                structure_id = nmt.bs.dict_to_selected[structure_id]
                cur_region = nmt.bs.level.loc[structure_id, "Abbrevation"]
            else:
                cur_region = "unknown"
            cur_region = pd.DataFrame({"Hemisphere":[cur_hemi],
                                       "Region":[cur_region]},
                                      index=[cur_name])
            self.region = pd.concat([self.region, cur_region])
        self.normalize()
        return

    def load_csv_from_path(self, path):
        self.csv_path = path
        self.metadata = pd.DataFrame(columns=['File_name'])
        for input_table in sorted(os.listdir(path)):
            cell_name = input_table.replace(".csv", "").replace(".swc", "").replace(".eswc", "")
            # 1. read a table
            if not input_table.endswith(".csv"):
                continue
            input_table = os.path.join(path, input_table)
            df = pd.read_csv(input_table)
            df.index = [cell_name]

            # 1. Feature (XYZ location) table
            self.raw_data = pd.concat([self.raw_data, df[["x", "y", "z"]]])
            scale_data = [int(df["x"] / nmt.annotation.space["x"]),
                          int(df["y"] / nmt.annotation.space["y"]),
                          int(df["z"] / nmt.annotation.space["z"])
                          ]

            # 2. Metadata
            cur_metadata = pd.DataFrame({'File_name':os.path.join(path, input_table)},
                                        index=[cell_name])
            self.metadata = pd.concat([self.metadata, cur_metadata])

            # 3. Region_table
            if df.z[0] < (nmt.annotation.micron_size['z']/2):
                cur_hemi = 1
            else:
                cur_hemi = 2
            structure_id = nmt.annotation.array[scale_data[0], scale_data[1], scale_data[2]]
            if structure_id in nmt.bs.level.index.tolist():
                cur_region = nmt.bs.level.loc[structure_id, "Abbrevation"]
            else:
                cur_region = "unknown"
            cur_region = pd.DataFrame({"Hemisphere":[cur_hemi], "Region":[cur_region]}, index=[cell_name])
            self.region = pd.concat([self.region, cur_region])
        return
    def normalize(self):
        scaled_data = scale(self.raw_data)
        self.scaled_data = pd.DataFrame(scaled_data,
                                        index=self.raw_data.index,
                                        columns=self.raw_data.columns)
        return

class dendrite_features(features):
    def __init__(self):
        features.__init__(self, "Dendrite")
        self.scaled_data = pd.DataFrame()
        return

    # def load_csv_from_path(self, path):
    #     self.csv_path = path
    #     self.metadata = pd.DataFrame(columns=['File_name', 'Hemisphere'])
    #     hemi_dict = {1: initiate_df_dict(), 2: initiate_df_dict()}
    #     for input_table in sorted(os.listdir(path)):
    #         df = read_location_table(path, input_table)
    #         if df is None:
    #             continue
    #         cell_name = input_table.replace(".csv", "").replace(".swc", "").replace(".eswc", "")
    #
    #         # Metadata
    #         hemi_id = get_hemisphere(df)
    #         if np.isnan(hemi_id):
    #             print(input_table)
    #             continue
    #         cur_metadata = pd.DataFrame({'File_name':os.path.join(path, input_table),
    #                                      'Hemisphere':hemi_id
    #                                      },
    #                                     index=[cell_name])
    #         self.metadata = pd.concat([self.metadata, cur_metadata])
    #
    #         # Feature table
    #         for i in [1,2]:
    #             hemi_dict[i] = add_new_record(hemi_dict[i], df, i, cell_name)
    #
    #     # ipsi_col = ["ipsi_" + nmt.bs.level.loc[i, "Abbrevation"] for i in hemi_dict[1]["soma"].columns.tolist()]
    #     ipsi_col = ["ipsi_" + nmt.bs.level.loc[i, "Abbrevation"] for i in hemi_dict[1]["axon"].columns.tolist()]
    #     contra_col = ["contra_" + nmt.bs.level.loc[i, "Abbrevation"] for i in hemi_dict[1]["axon"].columns.tolist()]
    #     axon_location = pd.DataFrame(index=hemi_dict[1]["axon"].index, columns=ipsi_col + contra_col, dtype='float32')
    #     # Flip the matrix if hemisphere == 2
    #     for i in range(len(self.metadata)):
    #         cell = self.metadata.index[i]
    #         hemi = self.metadata.Hemisphere[i]
    #         if hemi == 1:
    #             axon_location.loc[cell] = hemi_dict[1]["axon"].loc[cell].tolist() + \
    #                                       hemi_dict[2]["axon"].loc[cell].tolist()
    #         if hemi == 2:
    #             axon_location.loc[cell] = hemi_dict[2]["axon"].loc[cell].tolist() + \
    #                                       hemi_dict[1]["axon"].loc[cell].tolist()
    #     self.add_raw_data(axon_location)
    #     return

    def load_data_from_neuron_dict(self, neuron_dict):
        assert type(neuron_dict) == dict, "Error: projection_features.load_data_from_neuron_dict(self, neuron_dict).\nneuron_dict provided is NOT a python dictionary."
        for cur_name, cur_neuron in neuron_dict.items():
            assert type(cur_neuron) == neuron, "Error: projection_features.load_data_from_neuron_dict(self, neuron_dict).\nvalue of neuron_dict is NOT a nmt.neuron."
        region_used = nmt.bs.selected_regions
        columns = [nmt.bs.level.loc[i, "Abbrevation"] for i in region_used]

        df = pd.DataFrame(columns=columns)
        # TODO: show progress...
        for cur_name, cur_neuron in list(neuron_dict.items()):
            cur_df = cur_neuron.get_region_matrix(annotation=nmt.annotation,
                                                  brain_structure=nmt.bs,
                                                  region_used=None)
            df.loc[cur_name] = (np.sum(cur_df.loc[cur_df["hemisphere_id"] == 1, ["(basal) dendrite", "apical dendrite"]], axis=1) + \
                                np.sum(cur_df.loc[cur_df["hemisphere_id"] == 2, ["(basal) dendrite", "apical dendrite"]], axis=1)).tolist()
        self.add_raw_data(df)
        self.normalize(log=True)
        return


    def normalize(self, log=True):
        scaled_data = np.array(self.raw_data) / np.sum(self.raw_data, axis=1).values.reshape(-1,1) * 100000
        if log:
            scaled_data = np.log(scaled_data+100)
        self.scaled_data = pd.DataFrame(scaled_data, index=self.raw_data.index, columns=self.raw_data.columns)
        return
