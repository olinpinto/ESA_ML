from sklearn.model_selection import train_test_split

# Normalising the weights by type for a given value
def normalize_weights(df):
    total_weights = df.groupby('Type')['weight'].transform('sum')
    df['normalized_weight'] = df['weight'] / total_weights * 100000
    return df

# Functionality for splitting the 
def ml_preprocessing(df_combined_ml, test_size=0.25, random_state=14):
    # Split data into training and test sets
    X_train_full, X_test_full, y_train, y_test = train_test_split(df_combined_ml, df_combined_ml['Type'], test_size=test_size, random_state=random_state)

    # Normalize weights for training set
    X_train_full = normalize_weights(X_train_full)
    # Normalize weights for test set
    X_test_full = normalize_weights(X_test_full)

    # Extracting weights for training and test sets
    train_weights_ml = X_train_full['normalized_weight'].tolist()
    test_weights_ml = X_test_full['normalized_weight'].tolist()

    # Creating training and testing dataframes without the 'diphotonMass' and other columns
    X_train = X_train_full.drop(columns=['Type', 'weight', 'normalized_weight', 'diphotonMass'])
    X_test = X_test_full.drop(columns=['Type', 'weight', 'normalized_weight', 'diphotonMass'])

    # Convert labels to int64
    y_train = y_train.astype('int64')
    y_test = y_test.astype('int64')

    return X_train, X_test, y_train, y_test, train_weights_ml, test_weights_ml, X_train_full, X_test_full


