import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os

def prepare_classification_data(file_path='campus_movement_logs.csv'):
    if not os.path.exists(file_path):
        return None, None, None, None
        
    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Sort
    df = df.sort_values(by=['User_ID', 'Timestamp'])
    
    # Sessionization (same logic as preprocessing.py)
    df['Time_Diff'] = df.groupby('User_ID')['Timestamp'].diff()
    session_threshold = pd.Timedelta(minutes=90)
    df['New_Session'] = (df['Time_Diff'].isnull()) | (df['Time_Diff'] > session_threshold)
    df['Session_ID'] = df.groupby('User_ID')['New_Session'].cumsum()
    df['Global_Session_ID'] = df['User_ID'].astype(str) + "_" + df['Session_ID'].astype(str)
    
    # Create Target: Next_Location
    df['Next_Location'] = df.groupby('Global_Session_ID')['Location_ID'].shift(-1)
    
    # Drop rows without a next location (the last movement in a session)
    df = df.dropna(subset=['Next_Location']).copy()
    
    # Feature extraction
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Current_Location'] = df['Location_ID']
    
    # Encoding categorical variable (Location)
    le_loc = LabelEncoder()
    # Fit on all possible locations to ensure consistent mapping
    all_locations = pd.concat([df['Current_Location'], df['Next_Location']]).unique()
    le_loc.fit(all_locations)
    
    df['Current_Location_Encoded'] = le_loc.transform(df['Current_Location'])
    df['Next_Location_Encoded'] = le_loc.transform(df['Next_Location'])
    
    # Select features and target
    X = df[['Current_Location_Encoded', 'Hour', 'DayOfWeek']]
    y = df['Next_Location_Encoded']
    
    return X, y, le_loc, all_locations

def train_and_evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Naïve Bayes Classifier
    nb_classifier = CategoricalNB()
    nb_classifier.fit(X_train, y_train)
    nb_pred = nb_classifier.predict(X_test)
    nb_accuracy = accuracy_score(y_test, nb_pred)
    
    # 2. K-Nearest Neighbors Classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(X_train, y_train)
    knn_pred = knn_classifier.predict(X_test)
    knn_accuracy = accuracy_score(y_test, knn_pred)
    
    models = {
        'Naive_Bayes': nb_classifier,
        'KNN': knn_classifier
    }
    accuracies = {
        'Naive_Bayes': nb_accuracy,
        'KNN': knn_accuracy
    }
    
    return models, accuracies

def predict_next_location(model, le_loc, current_loc, hour, day_of_week):
    loc_encoded = le_loc.transform([current_loc])[0]
    # Use DataFrame to avoid scikit-learn feature name warnings
    input_data = pd.DataFrame([[loc_encoded, hour, day_of_week]], columns=['Current_Location_Encoded', 'Hour', 'DayOfWeek'])
    pred_encoded = model.predict(input_data)[0]
    return le_loc.inverse_transform([pred_encoded])[0]

if __name__ == '__main__':
    print("Loading data and preparing features...")
    X, y, le_loc, all_locs = prepare_classification_data()
    if X is not None:
        print("Training Naïve Bayes and KNN models...")
        models, accuracies = train_and_evaluate_models(X, y)
        print(f"Naïve Bayes Accuracy: {accuracies['Naive_Bayes']:.2%}")
        print(f"KNN Accuracy: {accuracies['KNN']:.2%}")
        
        # Test prediction
        test_loc = 'Dorms'
        test_hour = 8
        test_day = 0 # Monday
        print(f"\nPrediction for someone at {test_loc} at {test_hour}:00 on Monday:")
        nb_pred = predict_next_location(models['Naive_Bayes'], le_loc, test_loc, test_hour, test_day)
        knn_pred = predict_next_location(models['KNN'], le_loc, test_loc, test_hour, test_day)
        print(f"  Naïve Bayes predicts: {nb_pred}")
        print(f"  KNN predicts: {knn_pred}")
    else:
        print("Dataset not found. Generate data first.")
