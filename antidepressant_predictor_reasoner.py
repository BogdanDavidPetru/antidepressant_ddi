class MCSReasoner:
    def __init__(self, model, common_mcs_list, test_input, column_names, output_file_name):
        self.model = model
        self.test_input = test_input
        self.column_names = column_names
        self.common_mcs_list = common_mcs_list
        self.output_file_name = output_file_name

    def analyze_decision_paths(self):
        output_column_names = self.column_names
        input_features = self.test_input#.to_numpy()

        with open(self.output_file_name, 'w') as fp:
            for index, name in enumerate(output_column_names):
                node_indicator = self.model.estimators_[index].decision_path(self.test_input)
                leave_id = self.model.estimators_[index].apply(self.test_input)
                feature = self.model.estimators_[index].tree_.feature
                threshold = self.model.estimators_[index].tree_.threshold
                node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

                fp.write(f'Decision path for {name}:\n')

                for node_id in node_index:
                    if leave_id[0] == node_id:
                        fp.write(f'Leaf node {node_id} reached\n')
                        break

                    if input_features[0, feature[node_id]] < threshold[node_id]:
                        contains = "does not contain"
                    else:
                        contains = "contains"

                    feature_nr = feature[node_id]
                    if feature_nr >= len(self.common_mcs_list) / 2:
                        index_feature = feature_nr - len(self.common_mcs_list)
                        if index_feature >= len(self.common_mcs_list):
                            print(f"Error: Invalid index {index_feature}")
                        else:
                            fp.write(f'Second drug {contains} a chemical structure like: {self.common_mcs_list[index_feature]}\n')
                    else:
                        fp.write(f'First drug {contains} a chemical structure like: {self.common_mcs_list[feature_nr]}\n')

# Usage Example:

# Assuming you have the necessary inputs: mcs_file, model, test_input, and test_output
# mcs_file = "used_mcs_with_index.txt"
# model = trained_multilabel_decision_tree_model
# test_input = your_test_input_dataframe
# test_output = your_test_output_dataframe

# Initialize the MCSProcessor
# processor = MCSProcessor(mcs_file, model, test_input, test_output)

# Load the MCS data
# processor.load_mcs()

# Analyze decision paths
# processor.analyze_decision_paths()
