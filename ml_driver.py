import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

class_labels = {
    0: 'ggH',
    1: 'VBFlike',
    2: 'ttH',
    3: 'Rest',
    4: 'Bkg'
}

class_colors = {
    'ggH': 'blue',
    'VBFlike': 'orange',
    'ttH': 'purple',
    'Rest': 'lightgreen',
    'Bkg': 'black'
}

class Net(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], hidden_sizes[4])
        self.fc6 = nn.Linear(hidden_sizes[4], hidden_sizes[5])
        self.fc7 = nn.Linear(hidden_sizes[5], hidden_sizes[6])
        self.fc8 = nn.Linear(hidden_sizes[6], hidden_sizes[7])
        self.fc9 = nn.Linear(hidden_sizes[7], output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        out = self.relu(out)
        out = self.fc6(out)
        out = self.relu(out)
        out = self.fc7(out)
        out = self.relu(out)
        out = self.fc8(out)
        out = self.relu(out)
        out = self.fc9(out)
        return out
    
def get_network_parameters():
    input_size = 70 
    hidden_sizes = [1024, 512, 256, 128, 64, 32, 16, 8] 
    output_size = 5  
    learning_rate = 0.01
    num_epochs = 10
    batch_size = 64
    return input_size, hidden_sizes, output_size, learning_rate, num_epochs, batch_size

def initialize_network(input_size, hidden_sizes, output_size, learning_rate):
    net = Net(input_size, hidden_sizes, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    return net, criterion, optimizer

def normalize_data(train_data, test_data):
    scaler = StandardScaler()
    train_data_normalized = scaler.fit_transform(train_data)
    test_data_normalized = scaler.transform(test_data)
    return train_data_normalized, test_data_normalized

def create_dataloaders(train_data, train_weights, train_labels, test_data, test_weights, test_labels, batch_size):
    train_data = train_data.to_numpy() if isinstance(train_data, pd.DataFrame) else train_data
    test_data = test_data.to_numpy() if isinstance(test_data, pd.DataFrame) else test_data

    train_labels = train_labels.to_numpy() if isinstance(train_labels, pd.Series) else train_labels
    test_labels = test_labels.to_numpy() if isinstance(test_labels, pd.Series) else test_labels

    train_tensor = torch.tensor(train_data, dtype=torch.float)
    train_weight_tensor = torch.tensor(train_weights, dtype=torch.float)
    train_label_tensor = torch.tensor(train_labels.ravel().astype(int), dtype=torch.long)

    test_tensor = torch.tensor(test_data, dtype=torch.float)
    test_weight_tensor = torch.tensor(test_weights, dtype=torch.float)
    test_label_tensor = torch.tensor(test_labels.ravel().astype(int), dtype=torch.long)

    train_set = torch.utils.data.TensorDataset(train_tensor, train_label_tensor, train_weight_tensor)
    test_set = torch.utils.data.TensorDataset(test_tensor, test_label_tensor, test_weight_tensor)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train_and_evaluate(net, criterion, optimizer, train_loader, test_loader, num_epochs, column_names):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    predicted_probabilities = defaultdict(lambda: defaultdict(list))

    test_data_list = []
    test_labels_list = []
    test_predicted_list = []
    test_weights_list = []

    for epoch in range(num_epochs):
    
        test_data_list = []
        test_labels_list = []
        test_predicted_list = []
        test_weights_list = []
        
        net.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for i, (inputs, labels, weights) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            weighted_loss = torch.mean(weights * loss)
            weighted_loss.backward()
            optimizer.step()
            running_loss += weighted_loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_losses.append(running_loss / (i+1))
        train_accuracies.append(100 * correct_train / total_train)

        net.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for i, (inputs, labels, weights) in enumerate(test_loader):
                outputs = net(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                loss = criterion(outputs, labels)
                weighted_loss = torch.mean(weights * loss)
                test_loss += weighted_loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                
                if epoch == num_epochs - 1:
                    test_data_list.append(inputs)
                    test_labels_list.append(labels)
                    test_predicted_list.append(predicted)
                    test_weights_list.append(weights)

                for p, t, probs in zip(predicted, labels, probabilities):
                    predicted_probabilities[p.item()][t.item()].append(probs[p].item())
                    
        # Diagnostic to check which classes are actually predicted
        pred_counts = Counter([p.item() for p in predicted])
        #print("Predicted class counts:", {class_labels[k]: v for k, v in pred_counts.items()})

        test_losses.append(test_loss / (i+1))
        test_accuracies.append(100 * correct_test / total_test)

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_losses[-1]:.4f}, Testing Loss: {test_losses[-1]:.4f}, Training Accuracy: {train_accuracies[-1]:.2f}%, Testing Accuracy: {test_accuracies[-1]:.2f}%')

    if len(test_data_list) > 0:
        test_data = torch.vstack(test_data_list).numpy()
        test_labels = torch.hstack(test_labels_list).numpy()
        test_predicted = torch.hstack(test_predicted_list).numpy()
        test_weights = torch.hstack(test_weights_list).numpy()

        df_test_results = pd.DataFrame(test_data, columns=column_names)
        df_test_results['true_label'] = test_labels
        df_test_results['predicted_label'] = test_predicted
        df_test_results['weight'] = test_weights
    else:
        df_test_results = None

    return train_losses, test_losses, train_accuracies, test_accuracies, predicted_probabilities, df_test_results

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_model(filepath, input_size, hidden_sizes, output_size):
    net = Net(input_size, hidden_sizes, output_size)
    net.load_state_dict(torch.load(filepath))
    net.eval() 
    return net

def make_predictions(model, data):
    data_tensor = torch.tensor(data, dtype=torch.float)
    with torch.no_grad():
        outputs = model(data_tensor)
    return outputs

def aggregate_test_results(test_data_list, test_labels_list, test_predicted_list, test_weights_list, column_names):
    if len(test_data_list) > 0:
        test_data = torch.vstack(test_data_list).numpy()
        test_labels = torch.hstack(test_labels_list).numpy()
        test_predicted = torch.hstack(test_predicted_list).numpy()
        test_weights = torch.hstack(test_weights_list).numpy()

        df_test_results = pd.DataFrame(test_data, columns=column_names)
        df_test_results['true_label'] = test_labels
        df_test_results['predicted_label'] = test_predicted
        df_test_results['weight'] = test_weights
        return df_test_results
    return None

def calculate_bin_edges(predicted_probabilities, num_bins):
    all_probs = []
    for pred_class, truth_class_probs in predicted_probabilities.items():
        for truth_class, prob_list in truth_class_probs.items():
            all_probs += prob_list
    return np.histogram_bin_edges(all_probs, bins=num_bins)

def evaluate_model(net, test_loader, column_names):
    net.eval() 
    test_losses = []
    test_accuracies = []
    predicted_probabilities = defaultdict(lambda: defaultdict(list))

    criterion = nn.CrossEntropyLoss()

    test_data_list = []
    test_labels_list = []
    test_predicted_list = []
    test_weights_list = []
    
    correct_test = 0
    total_test = 0
    test_loss = 0.0
    
    with torch.no_grad():
        for i, (inputs, labels, weights) in enumerate(test_loader):
            outputs = net(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            weighted_loss = torch.mean(weights * loss)
            test_loss += weighted_loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

            test_data_list.append(inputs)
            test_labels_list.append(labels)
            test_predicted_list.append(predicted)
            test_weights_list.append(weights)

            for p, t, probs in zip(predicted, labels, probabilities):
                predicted_probabilities[p.item()][t.item()].append(probs[p].item())

    df_test_results = pd.DataFrame(torch.vstack(test_data_list).numpy(), columns=column_names)
    df_test_results['true_label'] = torch.hstack(test_labels_list).numpy()
    df_test_results['predicted_label'] = torch.hstack(test_predicted_list).numpy()
    df_test_results['weight'] = torch.hstack(test_weights_list).numpy()

    test_accuracy = 100 * correct_test / total_test
    average_test_loss = test_loss / (i + 1)

    print(f"Testing Loss: {average_test_loss:.4f}, Testing Accuracy: {test_accuracy:.2f}%")

    return predicted_probabilities, df_test_results

def calculate_confusion_matrix(true_labels, predicted_labels, num_classes):
    confusion_matrix = torch.zeros(num_classes, num_classes)
    for t, p in zip(true_labels, predicted_labels):
        confusion_matrix[t, p] += 1
    return confusion_matrix