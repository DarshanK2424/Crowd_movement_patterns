import pandas as pd
from datetime import timedelta

def process_logs_to_transactions(input_file='campus_movement_logs.csv', output_file='transactions.csv'):
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Sort by user and then timestamp
    df = df.sort_values(by=['User_ID', 'Timestamp'])
    
    # Calculate time difference between consecutive pings for the same user
    df['Time_Diff'] = df.groupby('User_ID')['Timestamp'].diff()
    
    # Define a session: A new session starts if the time diff is greater than 90 minutes
    # (Since our data generator adds 10-60 mins between locations in the same session)
    session_threshold = pd.Timedelta(minutes=90)
    
    # True if new session
    df['New_Session'] = (df['Time_Diff'].isnull()) | (df['Time_Diff'] > session_threshold)
    
    # Cumulative sum creates a unique session ID per user
    df['Session_ID'] = df.groupby('User_ID')['New_Session'].cumsum()
    df['Global_Session_ID'] = df['User_ID'].astype(str) + "_" + df['Session_ID'].astype(str)
    
    # Group by the global session ID to get the list of locations (transaction)
    # Maintain order of locations
    transactions_df = df.groupby('Global_Session_ID')['Location_ID'].apply(list).reset_index(name='Transaction')
    
    # Filter out sessions with only 1 location (no pattern to mine)
    transactions_df = transactions_df[transactions_df['Transaction'].apply(len) > 1]
    
    print(f"Extracted {len(transactions_df)} valid movement sessions (transactions) from raw logs.")
    
    # Save transactions
    # Format: Loc1,Loc2,Loc3...
    transactions_df['Transaction_Str'] = transactions_df['Transaction'].apply(lambda x: ','.join(x))
    transactions_df[['Global_Session_ID', 'Transaction_Str']].to_csv(output_file, index=False)
    
    return transactions_df['Transaction'].tolist()

if __name__ == '__main__':
    transactions = process_logs_to_transactions()
    print("Sample transactions:")
    for t in transactions[:5]:
        print(t)
