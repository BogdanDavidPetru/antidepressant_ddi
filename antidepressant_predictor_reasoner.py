class MCSReasoner:
    def __init__(self, model, common_mcs_list, test_input, column_names, output_file_name):
        self.model = model
        self.test_input = test_input
        self.column_names = column_names
        self.common_mcs_list = common_mcs_list
        self.output_file_name = output_file_name

    def analyze_decision_paths(self):
        output_column_names = self.column_names
        input_features = self.test_input
        drug1_contains_dict = {}
        drug1_not_contains_dict = {}
        drug2_contains_dict = {}
        drug2_not_contains_dict = {}

        with open(self.output_file_name, 'w') as fp:
            fp.write(f'Decision path for binary interaction\n')
            for index, tree in enumerate(self.model.estimators_):
                node_indicator = tree.decision_path(self.test_input)
                feature_importance_list = tree.feature_importances_
                leave_id = tree.apply(self.test_input)
                feature = tree.tree_.feature
                threshold = tree.tree_.threshold
                node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

                for node_id in node_index:
                    if leave_id[0] == node_id:
                        # fp.write(f'Leaf node {node_id} reached\n')
                        break
                    if input_features[0, feature[node_id]] <= threshold[node_id]:
                        contains = False
                        # contains = "does not contain"
                    else:
                        contains = True
                        # contains = "contains"

                    feature_nr = feature[node_id]
                    feature_importance_nr = feature_importance_list[feature_nr]
                    if feature_nr >= len(self.common_mcs_list) / 2:
                        index_feature = feature_nr - len(self.common_mcs_list)
                        if index_feature >= len(self.common_mcs_list):
                            print(f"Error: Invalid index {index_feature}")
                        else:
                            mcs = self.common_mcs_list[index_feature]
                            if contains:
                                if mcs in drug2_contains_dict:
                                    drug2_contains_dict[mcs] += feature_importance_nr
                                else:
                                    drug2_contains_dict[mcs] = feature_importance_nr
                            else:
                                if mcs in drug2_not_contains_dict:
                                    drug2_not_contains_dict[mcs] += feature_importance_nr
                                else:
                                    drug2_not_contains_dict[mcs] = feature_importance_nr
                            # fp.write(f'Second drug {contains} a chemical structure like: {self.common_mcs_list[index_feature]}\n')
                    else:
                        mcs = self.common_mcs_list[feature_nr]
                        if contains:
                            if mcs in drug1_contains_dict:
                                drug1_contains_dict[mcs] += feature_importance_nr
                            else:
                                drug1_contains_dict[mcs] = feature_importance_nr
                        else:
                            if mcs in drug1_not_contains_dict:
                                drug1_not_contains_dict[mcs] += feature_importance_nr
                            else:
                                drug1_not_contains_dict[mcs] = feature_importance_nr
                    # break
                        # fp.write(f'First drug {contains} a chemical structure like: {self.common_mcs_list[feature_nr]}\n')
            drug1_contains_dict_sorted = dict(sorted(drug1_contains_dict.items(), key=lambda item: item[1], reverse=True))
            drug1_not_contains_dict_sorted = dict(sorted(drug1_not_contains_dict.items(), key=lambda item: item[1], reverse=True))
            drug2_contains_dict_sorted = dict(sorted(drug2_contains_dict.items(), key=lambda item: item[1], reverse=True))
            drug2_not_contains_dict_sorted = dict(sorted(drug2_not_contains_dict.items(), key=lambda item: item[1], reverse=True))

            fp.write('\n-------------Drug 1 structures influencing decision------------------\n')
            for mcs, importance in drug1_contains_dict_sorted.items():
                fp.write(f'Importance {importance} : 1st drug contains a chemical structure like: {mcs}\n')
            for mcs, importance in drug1_not_contains_dict_sorted.items():
                fp.write(f'Importance {importance} : 1st drug does not contain a chemical structure like: {mcs}\n')

            fp.write('\n-------------Drug 2 structures influencing decision------------------\n')
            for mcs, importance in drug2_contains_dict_sorted.items():
                fp.write(f'Importance {importance} : 2nd drug contains a chemical structure like: {mcs}\n')
            for mcs, importance in drug2_not_contains_dict_sorted.items():
                fp.write(f'Importance {importance} : 2nd drug does not contain a chemical structure like: {mcs}\n')

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
