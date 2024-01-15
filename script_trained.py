import data_loader as dl
import preprocessor as pp
import ml_preprocessor as mlp
import ml_driver as mld
import analyser as al
import plotter as pt

from ml_driver import class_colors, class_labels
from analyser import mu_values
import numpy as np

# IMPORTING
# Data imports with local paths
ggH_df = dl.load_data("LOCAL_PATH/ggH_M125_processed.parquet")
VBF_df = dl.load_data("LOCAL_PATH/VBF_M125_processed.parquet")
VH_df = dl.load_data("LOCAL_PATH/VH_M125_processed.parquet")
ttH_df = dl.load_data("LOCAL_PATH/ttH_M125_processed.parquet")
bkg1_df = dl.load_data("LOCAL_PATH/GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_processed.parquet")
bkg2_df = dl.load_data("LOCAL_PATH/GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_processed.parquet")
bkg3_df = dl.load_data("LOCAL_PATH/DiPhotonJetsBox_M40_80_processed.parquet")
bkg4_df = dl.load_data("LOCAL_PATH/DiPhotonJetsBox_MGG-80toInf_processed.parquet")
print('Loading completed!')

# PROCESSING
# Data preprocessing - initial tagging to differentiate between the different types and create 'Type' column
ggH_red_df = pp.preprocess_dataframe(ggH_df, 11)
VBF_red_df = pp.preprocess_dataframe(VBF_df, 10)
VH_red_df = pp.preprocess_dataframe(VH_df, 13)
ttH_red_df = pp.preprocess_dataframe(ttH_df, 12)
bkg1_red_df = pp.preprocess_dataframe(bkg1_df, 4)
bkg2_red_df = pp.preprocess_dataframe(bkg2_df, 4)
bkg3_red_df = pp.preprocess_dataframe(bkg3_df, 4)
bkg4_red_df = pp.preprocess_dataframe(bkg4_df, 4)
print('Preprocessing copleted!')

# Combining the preprocessed background and signal dataframes together
combined_df = pp.combine_and_preprocess_dfs(
    [ggH_red_df, VBF_red_df, VH_red_df, ttH_red_df], 
    [bkg1_red_df, bkg2_red_df, bkg3_red_df, bkg4_red_df]
)
print('Combining completed!')

# Prepare the dataframe for machine learning applications - splitting, extracting normalised weights, retaining a copy of full frame
X_train, X_test, y_train, y_test, train_weights_ml, test_weights_ml, X_train_full, X_test_full = mlp.ml_preprocessing(combined_df)
print("Check - sum of normalised train weights:", sum(train_weights_ml))
print("Check - sum of normalised test weights:", sum(test_weights_ml))
print("Check - unique 'Type' labels in the training set:", y_train.unique())
print("Check - unique 'Type' labels in the test set:", y_test.unique())
print("Splitting completed!")

# MACHINE LEARNING
# Fetching network parameters (modifiable from the functions in ml_driver.py module)
input_size, hidden_sizes, output_size, learning_rate, num_epochs, batch_size = mld.get_network_parameters()

# Initialising, training and loading the ML model
model_filepath = 'trained_model.pth'
net, criterion, optimizer = mld.initialize_network(input_size, hidden_sizes, output_size, learning_rate)
net = mld.load_model(model_filepath, input_size, hidden_sizes, output_size)

# Normalising the data
train_data_normalized, test_data_normalized = mld.normalize_data(X_train, X_test)

# Creating DataLoaders
train_loader, test_loader = mld.create_dataloaders(train_data_normalized, train_weights_ml, y_train,
                                                       test_data_normalized, test_weights_ml, y_test, batch_size)

# Fitting, fetching column names
print('Fitting started!')
column_names = X_train.columns
predicted_probabilities, df_test_results = mld.evaluate_model(net, test_loader, column_names)
print('Fitting completed!')

# Add the original weights and diphotonMasses to the dataframe for further reference
df_test_results_with_mass = pp.add_diphotonMass_to_results(df_test_results, X_test_full)

# PLOTTING
# Raw data histograms by true label
print('Plotting started!')
pt.plot_histograms_by_class(df_test_results.drop(['true_label', 'predicted_label', 'weight'], axis=1), 
                         df_test_results['true_label'], 
                         column_names,
                         50,
                         'histograms_by_class.pdf')

# Calculating the bin edges for probability plotting purposes, plotting probability curves
bin_edges = mld.calculate_bin_edges(predicted_probabilities, 20)
pt.plot_probability_curves(predicted_probabilities, bin_edges, 'probability_curves')

# Plotting the ROC and Efficiency curves for all the data combinations
for class_idx in class_labels.keys():
    pt.plot_efficiency_curves(predicted_probabilities, f'efficiency_curves_{class_labels[class_idx]}', class_idx)

for class_idx in class_labels.keys():
    pt.plot_roc_curves(predicted_probabilities, f'roc_curves_{class_labels[class_idx]}', class_idx)

# Calculate and plotting the confusion matrix
confusion_matrix = mld.calculate_confusion_matrix(df_test_results['true_label'], df_test_results['predicted_label'], len(class_labels))
pt.plot_confusion_matrix(df_test_results['true_label'], df_test_results['predicted_label'], 'confusion_matrix')

# Plotting the diphoton mass distribution
pt.plot_diphoton_mass(df_test_results_with_mass, 'mass_plot', 50, 100, 180,)

# Plotting feature distributions by predicted labels
pt.plot_feature_distributions_by_predicted_label(df_test_results, 'dist_predicted_label')

print('Initial plotting completed!')

# LIKELIHOOD CALCULATIONS
# Calculating and plotting Poisson likelihoods per class, setting mu values at 1 for all the classes
print('Starting Poisson calculation')
likelihood_results = al.calculate_poisson_likelihood(df_test_results_with_mass, mu_values)
pt.plot_likelihood_curves(likelihood_results, 'poisson_likelihood')

# Calculating Profiled likelihoods
print('Starting Profiled calculation')
signal_labels = [label for label, class_name in class_labels.items() if class_name != 'Bkg']
profiled_results = al.profile_likelihoods_for_all_signals(df_test_results_with_mass, np.arange(-2, 5, 0.5), (100, 180), 80, signal_labels, mu_values)
pt.plot_profiled_likelihood_curves(profiled_results, class_labels, 'likelihood_curves')
pt.export_to_csv(profiled_results, 'profiled_likelihoods.csv')