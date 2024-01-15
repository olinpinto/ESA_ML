import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

mu_values = {
    0: 1,
    1: 1,
    2: 1,
    3: 1
}

def calculate_single_likelihood(df, label, mu_values, mass_range, num_bins):
    neg_two_log_likelihood = 0
    df_label = df[df.predicted_label == label]
    bins = np.linspace(mass_range[0], mass_range[1], num_bins+1)
    df_label = df[df.predicted_label == label].copy() 
    df_label['bin'] = np.digitize(df_label['diphotonMass'], bins) - 1

    for bin_num in range(num_bins):
        df_bin = df_label[df_label.bin == bin_num]
        k = df_bin['scaled_original_weight'].sum()
        lambda_ = 0
        for true_label in df_bin.true_label.unique():
            sum_weights = df_bin[df_bin.true_label == true_label]['scaled_original_weight'].sum()
            mu_value = mu_values.get(true_label, 1)
            lambda_ += sum_weights * mu_value
        log_P = -lambda_ + k * np.log(lambda_) - gammaln(k + 1) if lambda_ > 0 else (0 if k == 0 else -np.inf)
        neg_two_log_likelihood -= 2 * log_P

    return neg_two_log_likelihood

def calculate_poisson_likelihood(df, mu_values, mass_range=(100, 180), num_bins=80):
    df_filtered = df[(df['diphotonMass'] >= mass_range[0]) & (df['diphotonMass'] <= mass_range[1])].copy()
    likelihood_results = {}

    for label in df_filtered['predicted_label'].unique():
        mu_values_plot = np.arange(-2, 4.1, 0.1)
        neg_two_log_likelihoods = [calculate_single_likelihood(df_filtered, label, {**mu_values, label: mu}, mass_range, num_bins) for mu in mu_values_plot]

        min_likelihood = min(neg_two_log_likelihoods)
        neg_two_log_likelihoods = [x - min_likelihood for x in neg_two_log_likelihoods]
        likelihood_results[label] = (mu_values_plot, neg_two_log_likelihoods)

    return likelihood_results

def calculate_profiled_likelihood(df, fixed_label, fixed_mu, mu_values, mass_range, num_bins, signal_labels):
    def likelihood_to_minimize(varying_mu_values):
        mu_values_combined = mu_values.copy()
        mu_values_combined[fixed_label] = fixed_mu
        varying_mu_index = 0
        for label in signal_labels:
            if label != fixed_label:
                mu_values_combined[label] = varying_mu_values[varying_mu_index]
                varying_mu_index += 1
        return calculate_single_likelihood(df, fixed_label, mu_values_combined, mass_range, num_bins)

    other_class_bounds = [(None, None) for _ in signal_labels if _ != fixed_label]
    initial_guesses = [mu_values[label] for label in signal_labels if label != fixed_label]
    result = minimize(likelihood_to_minimize, initial_guesses, bounds=other_class_bounds)
    return result.fun, result.x


def profile_likelihoods_for_all_signals(df, mu_range, mass_range, num_bins, signal_labels, mu_values):
    results = {}
    for label in signal_labels:
        for mu in mu_range:
            minimized_likelihood, profiled_mus = calculate_profiled_likelihood(df, label, mu, mu_values, mass_range, num_bins, signal_labels)
            results[(label, mu)] = (minimized_likelihood, profiled_mus)
    return results
