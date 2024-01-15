from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from ml_driver import class_colors, class_labels, calculate_confusion_matrix

def plot_histograms_by_class(data, labels, column_names, num_bins, filename, title_prefix='', max_rows_per_fig=10):
    num_cols = len(column_names)
    num_figs = np.ceil(num_cols / max_rows_per_fig)
    
    with PdfPages(f"LOCAL_PATH/{filename}.pdf") as pdf:
        for fig_idx in range(int(num_figs)):
            start_idx = fig_idx * max_rows_per_fig
            end_idx = min(start_idx + max_rows_per_fig, num_cols)
            fig_cols = column_names[start_idx:end_idx]

            fig, axes = plt.subplots(nrows=len(fig_cols), ncols=1, figsize=(10, 5*len(fig_cols)))
            fig.tight_layout(pad=5.0)

            for ax, col in zip(axes, fig_cols):
                bins = np.histogram_bin_edges(data[col].dropna(), bins=num_bins, range=(data[col].min(), data[col].max()))
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                for label, label_data in data.groupby(labels):
                    finite_label_data = label_data[col].replace([np.inf, -np.inf], np.nan).dropna()
                    ax.hist(finite_label_data, bins=bins, alpha=0.75, label=class_labels[int(label)], density=True, histtype='step', linewidth=2, color=class_colors[class_labels[int(label)]], align='mid')

                ax.set_title(f'{title_prefix}: {col}')
                ax.legend()

            pdf.savefig(fig)
            plt.close(fig)

def plot_training_testing_performance(train_losses, test_losses, train_accuracies, test_accuracies, num_epochs, filename):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), test_losses, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Performance')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, num_epochs+1), test_accuracies, label='Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(f"LOCAL_PATH/{filename}.pdf")
    plt.close()

def plot_probability_curves(predicted_probabilities, bin_edges, filename):
    plt.figure(figsize=(15, 15))
    num_classes = len(class_labels)
    fig, axes = plt.subplots(num_classes, 1, figsize=(10, 5*num_classes))
    axes = axes.flatten()

    for pred_class, ax in enumerate(axes):
        for truth_class, prob_list in predicted_probabilities[pred_class].items():
            ax.hist(prob_list, bins=bin_edges, alpha=0.75, weights=np.ones(len(prob_list)) / len(prob_list), label=f'{class_labels[truth_class]}', histtype='step', linewidth=2, color=class_colors[class_labels[truth_class]])

        ax.set_xlabel('Probability')
        ax.set_ylabel('Fraction')
        ax.set_title(f'Predicted Class {class_labels[pred_class]}')
        ax.legend(loc='upper center')

    plt.tight_layout()
    plt.savefig(f"LOCAL_PATH/{filename}.pdf")
    plt.close()

def plot_efficiency_curves(predicted_probabilities, filename, class_idx):
    plt.figure(figsize=(15, 15))
    combinations_list = [(class_idx, other_class) for other_class in class_labels.keys() if class_idx != other_class]
    n_plots = len(combinations_list)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))

    for idx, (class1, class2) in enumerate(combinations_list, start=1):
        ax = plt.subplot(n_rows, n_cols, idx)
        true_labels = []
        scores = []

        for truth_class in [class1, class2]:
            prob_list = predicted_probabilities[class1][truth_class]
            if truth_class == class1:
                true_labels += [1] * len(prob_list)
            elif truth_class == class2:
                true_labels += [0] * len(prob_list)
            scores += prob_list

        fpr, tpr, _ = roc_curve(true_labels, scores)
        signal_efficiency = tpr 
        background_efficiency = fpr

        ax.plot(signal_efficiency, background_efficiency, label=f'{class_labels[class1]} vs {class_labels[class2]}')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Signal Efficiency')
        ax.set_ylabel('Background Efficiency')
        ax.set_title(f'{class_labels[class1]} vs {class_labels[class2]}')
        ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(f"LOCAL_PATH/{filename}.pdf")
    plt.close()

def plot_roc_curves(predicted_probabilities, filename, class_idx):
    plt.figure(figsize=(10, 10))
    combinations_list = [(class_idx, other_class) for other_class in class_labels.keys() if class_idx != other_class]
    n_plots = len(combinations_list)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))

    for idx, (class1, class2) in enumerate(combinations_list, start=1):
        ax = plt.subplot(n_rows, n_cols, idx)
        true_labels = []
        scores = []
        
        for truth_class in [class1, class2]:
            prob_list = predicted_probabilities[class1][truth_class]
            if truth_class == class1:
                true_labels += [1] * len(prob_list)
            elif truth_class == class2:
                true_labels += [0] * len(prob_list)
            scores += prob_list

        fpr, tpr, _ = roc_curve(true_labels, scores)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, label=f'{class_labels[class1]} vs {class_labels[class2]} (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{class_labels[class1]} vs {class_labels[class2]}')
        ax.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(f"LOCAL_PATH/{filename}.pdf")
    plt.close()

def plot_confusion_matrix(true_labels, predicted_labels, filename):
    num_classes = len(class_labels)
    confusion_matrix = calculate_confusion_matrix(true_labels, predicted_labels, num_classes)

    row_sums = confusion_matrix.sum(axis=1)
    normalized_confusion_matrix = confusion_matrix / row_sums[:, None]

    plt.figure(figsize=(10, 8))
    sns.heatmap(normalized_confusion_matrix, annot=True, cmap='Blues', 
                xticklabels=class_labels.values(), 
                yticklabels=class_labels.values())
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix')
    plt.savefig(f"LOCAL_PATH/{filename}.pdf")
    plt.close()


def plot_feature_distributions_by_predicted_label(df_test_results, filename):
    with PdfPages(f"LOCAL_PATH/{filename}.pdf") as pdf_pages:
        predicted_labels = df_test_results['predicted_label'].unique()

        feature_names = df_test_results.columns.drop(['true_label', 'predicted_label', 'weight'])

        for feature_name in feature_names:
            plt.figure(figsize=(10, 6))
            bins = np.linspace(df_test_results[feature_name].min(), df_test_results[feature_name].max(), 101)
            for label in predicted_labels:
                subset = df_test_results[df_test_results['predicted_label'] == label]
                weights = np.ones_like(subset[feature_name])/float(len(subset[feature_name]))
                y, bin_edges = np.histogram(subset[feature_name], bins=bins, weights=weights)
                bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
                plt.plot(bin_centers, y, label=f'{class_labels[label]}', color=class_colors[class_labels[label]])

            plt.title(f'Plots of {feature_name} by Predicted Label')
            plt.xlabel(feature_name)
            plt.ylabel('Fraction')
            plt.legend()
            pdf_pages.savefig(plt.gcf())
            plt.close()


def plot_normalized_counts(df, filename, num_bins, min_mass, max_mass):
    plt.figure(figsize=(10,6))
    for label in df['predicted_label'].unique():
        label_data = df[(df['predicted_label'] == label) & (df['diphotonMass'] >= min_mass) & (df['diphotonMass'] <= max_mass)]
        weights = label_data['original_weight']
        counts, bin_edges = np.histogram(label_data['diphotonMass'], weights=weights, bins = num_bins)
        counts = counts / counts.sum()
        bin_width = bin_edges[1] - bin_edges[0]
        plt.bar(bin_edges[:-1], counts, width=bin_width, alpha=0.5,
                color=class_colors[class_labels[label]], label=class_labels[label])

    plt.xlabel('diphotonMass')
    plt.ylabel('Fraction of total events')
    plt.xlim(100, 180)
    plt.legend()
    plt.savefig(f"LOCAL_PATH/{filename}.pdf")
    plt.close()


def plot_normalized_counts_subplots(df, filename, num_bins, min_mass, max_mass):
    plt.figure(figsize=(10, 6))
    fig, axs = plt.subplots(len(class_labels), 1, figsize=(10, 6*len(class_labels)))
    for ax, label in zip(axs, df['predicted_label'].unique()):
        label_data = df[(df['predicted_label'] == label) & (df['diphotonMass'] >= min_mass) & (df['diphotonMass'] <= max_mass)]
        weights = label_data['original_weight']
        counts, bin_edges = np.histogram(label_data['diphotonMass'], weights=weights, bins = num_bins)
        counts = counts / counts.sum() 
        bin_width = bin_edges[1] - bin_edges[0]
        ax.bar(bin_edges[:-1], counts, width=bin_width, alpha=0.5,
            color=class_colors[class_labels[label]], label=class_labels[label])
        ax.set_xlabel('diphotonMass')
        ax.set_ylabel('Fraction of total events')
        ax.set_xlim(100, 180) 
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"LOCAL_PATH/{filename}.pdf") 
    plt.close()

def plot_total_event_counts(df, filename, num_bins, min_mass, max_mass):
    plt.figure(figsize=(10, 6))
    for label in df['predicted_label'].unique():
        label_data = df[(df['predicted_label'] == label) & (df['diphotonMass'] >= min_mass) & (df['diphotonMass'] <= max_mass)]
        plt.hist(label_data['diphotonMass'], bins = num_bins, alpha=0.5,
                color=class_colors[class_labels[label]], label=class_labels[label])

    plt.xlabel('diphotonMass')
    plt.ylabel('Total events')
    plt.xlim(100, 180)
    plt.legend()
    plt.savefig(f"LOCAL_PATH/{filename}.pdf")

def plot_total_event_counts_subplots(df, filename, num_bins, min_mass, max_mass):
    plt.figure(figsize=(10, 6))
    fig, axs = plt.subplots(len(class_labels), 1, figsize=(10, 6*len(class_labels)))

    for ax, label in zip(axs, df['predicted_label'].unique()):
        label_data = df[(df['predicted_label'] == label) & (df['diphotonMass'] >= min_mass) & (df['diphotonMass'] <= max_mass)]
        ax.hist(label_data['diphotonMass'], bins=num_bins, alpha=0.5,
                color=class_colors[class_labels[label]], label=class_labels[label])
        ax.set_xlabel('diphotonMass')
        ax.set_ylabel('Total events')
        ax.set_xlim(100, 180)
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"LOCAL_PATH/{filename}.pdf") 
    plt.close()


def plot_diphoton_mass(df, filename, num_bins, min_mass, max_mass):
    plt.figure(figsize=(10, 6))
    fig, axs = plt.subplots(len(class_labels), 1, figsize=(10, 6*len(class_labels)))
    for ax, label in zip(axs, df['predicted_label'].unique()):
        label_data = df[(df['predicted_label'] == label) & (df['diphotonMass'] >= min_mass) & (df['diphotonMass'] <= max_mass)]
        true_labels = label_data['true_label'].unique()
        data = [label_data[label_data['true_label'] == true_label]['diphotonMass'] for true_label in true_labels]
        weights = [label_data[label_data['true_label'] == true_label]['original_weight'] for true_label in true_labels]
        colors = [class_colors[class_labels[true_label]] for true_label in true_labels]
        labels = [class_labels[true_label] for true_label in true_labels]
        ax.hist(data, weights=weights, bins=num_bins, stacked=True, color=colors, label=labels, alpha=0.5)
        ax.set_xlabel('diphotonMass')
        ax.set_ylabel('Fraction of total events')
        ax.set_title(f'Predicted Label: {class_labels[label]}') 
        ax.set_xlim(100, 180)
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"LOCAL_PATH/{filename}.pdf") 
    plt.close()

def plot_likelihood_curves(likelihood_results, filename):
    with PdfPages(f"LOCAL_PATH/{filename}.pdf") as pdf:
        for label, (mu_values_plot, neg_two_log_likelihoods) in likelihood_results.items():
            plt.figure(figsize=(10, 6))
            plt.plot(mu_values_plot, neg_two_log_likelihoods, 'o-', label=class_labels.get(label, "Unknown"), color=class_colors.get(class_labels.get(label, 'grey')))
            plt.title(f"Likelihood curve for {class_labels.get(label, 'Unknown')}")
            plt.xlabel(r'$\mu$')
            plt.ylabel('-2 log likelihood')
            plt.grid(True)
            plt.ylim([0, 10])

            plt.axhline(y=1, color='r', linestyle='--')
            plt.axhline(y=4, color='r', linestyle='--')
            plt.text(-2, 1.1, r'$1\sigma$', va='bottom', ha="left")
            plt.text(-2, 4.1, r'$2\sigma$', va='bottom', ha="left")
            
            plt.legend()
            pdf.savefig()
            plt.close()

def export_to_csv(results, filename):
    with open(f"LOCAL_PATH/{filename}.csv", 'w') as f:
        f.write("Signal Class,Given Mu Value,Minimized Likelihood,Mu Values for Other Classes\n")
        for (label, mu), (likelihood, profiled_mus) in results.items():
            profiled_mu_str = ','.join(map(str, profiled_mus))
            f.write(f"{class_labels[label]},{mu},{likelihood},{profiled_mu_str}\n")

def plot_profiled_likelihood_curves(profiled_results, class_labels, filename):
    with PdfPages(f"LOCAL_PATH/{filename}.pdf") as pdf:
        for label in set(key[0] for key in profiled_results.keys()):
            filtered_results = {mu: likelihood for (lbl, mu), (likelihood, _) in profiled_results.items() if lbl == label}

            mu_values_plot = list(filtered_results.keys())
            neg_two_log_likelihoods = [filtered_results[mu] for mu in mu_values_plot]

            min_likelihood = min(neg_two_log_likelihoods)
            normalized_likelihoods = [l - min_likelihood for l in neg_two_log_likelihoods]
            max_normalized_likelihood = max(normalized_likelihoods)

            plt.figure(figsize=(10, 6))
            plt.plot(mu_values_plot, normalized_likelihoods, 'o-', label=class_labels.get(label, "Unknown"))
            plt.title(f"Normalized Profiled Likelihood Curve for {class_labels.get(label, 'Unknown')}")
            plt.xlabel(r'$\mu$')
            plt.ylabel('Normalized -2 log likelihood')
            plt.grid(True)
            plt.ylim([0, max_normalized_likelihood]) 
            plt.legend()
            pdf.savefig()
            plt.close()