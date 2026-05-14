import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_campus_data(num_users=500, days=7, output_file='campus_movement_logs.csv'):
    np.random.seed(42)
    random.seed(42)
    
    locations = ['Dorms', 'Cafeteria', 'CS_Building', 'Math_Building', 'Library', 'Student_Center', 'Gym']
    
    # Probabilities of transitioning from location A to location B
    # Rough logic to make frequent patterns emerge (e.g. Dorms -> Cafeteria -> CS_Building)
    transitions = {
        'Dorms': {'Cafeteria': 0.5, 'CS_Building': 0.2, 'Math_Building': 0.2, 'Library': 0.1},
        'Cafeteria': {'CS_Building': 0.4, 'Math_Building': 0.3, 'Library': 0.2, 'Student_Center': 0.1},
        'CS_Building': {'Library': 0.4, 'Student_Center': 0.3, 'Cafeteria': 0.2, 'Dorms': 0.1},
        'Math_Building': {'Library': 0.3, 'Student_Center': 0.3, 'Cafeteria': 0.2, 'Dorms': 0.2},
        'Library': {'Cafeteria': 0.3, 'Dorms': 0.4, 'Gym': 0.2, 'Student_Center': 0.1},
        'Student_Center': {'Dorms': 0.4, 'Library': 0.3, 'Gym': 0.3},
        'Gym': {'Dorms': 0.8, 'Cafeteria': 0.2}
    }
    
    records = []
    start_date = datetime(2026, 1, 1, 8, 0, 0)
    
    for user in range(num_users):
        user_id = f"U{user:04d}"
        for day in range(days):
            # Each user has 1 to 3 movement sessions per day
            num_sessions = random.randint(1, 3)
            current_time = start_date + timedelta(days=day)
            
            for _ in range(num_sessions):
                # Start a session
                current_loc = 'Dorms' if random.random() < 0.7 else random.choice(locations)
                session_length = random.randint(3, 8)
                
                # Offset start time randomly within the day
                current_time += timedelta(hours=random.uniform(0, 4))
                
                for step in range(session_length):
                    records.append({
                        'User_ID': user_id,
                        'Timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'Location_ID': current_loc
                    })
                    
                    # Advance time
                    current_time += timedelta(minutes=random.uniform(10, 60))
                    
                    # Next location
                    if current_loc in transitions:
                        next_locs = list(transitions[current_loc].keys())
                        probs = list(transitions[current_loc].values())
                        current_loc = np.random.choice(next_locs, p=probs)
                    else:
                        current_loc = random.choice(locations)
                        
    df = pd.DataFrame(records)
    # Sort chronologically to make it look like real streamed logs
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} movement records and saved to {output_file}")

if __name__ == "__main__":
    generate_campus_data()
