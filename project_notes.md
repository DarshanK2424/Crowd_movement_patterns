# Smart Campus Crowd Movement Analysis - Project Notes

## 🎯 What we are trying to solve/achieve
The goal of this project is to build an efficient data mining system to analyze crowd movement patterns on a "smart campus" using simulated WiFi connection logs. By analyzing the chronological locations of users, the project aims to:
1. **Discover Frequent Movement Patterns**: Understand which sequences of locations are commonly visited together by students (e.g., Dorms -> Cafeteria -> CS Building).
2. **Algorithm Comparison**: Compare the performance (execution time) of different Frequent Pattern Mining algorithms (Apriori vs. FP-Growth).
3. **Identify Campus Hotspots**: Determine the most central/popular locations on campus using network analysis techniques (PageRank).
4. **Identify Bottlenecks**: Find the heaviest traffic paths between locations.
5. **Predict Next Location (Classification)**: Use Supervised Learning algorithms (Naïve Bayes and K-Nearest Neighbors) to predict where a user will go next based on their current location, hour of the day, and day of the week.
This type of spatial-temporal analysis is crucial for effective campus resource management, scheduling, and infrastructure planning.

---

## 📁 File Explanations

Here is a breakdown of what each Python script in the project is doing:

### 1. `data_generator.py`
**Purpose**: Data Simulation
- This script generates synthetic campus WiFi movement logs (`campus_movement_logs.csv`).
- It simulates a set number of users moving between different campus locations (Dorms, Cafeteria, Library, etc.) over a specified number of days.
- It uses predefined "transition probabilities" to make certain movement patterns more likely to occur naturally in the generated dataset.

### 2. `preprocessing.py`
**Purpose**: Data Preparation & Sessionization
- It takes the raw chronologically sorted logs (`campus_movement_logs.csv`) and processes them into meaningful "sessions" or "transactions" (`transactions.csv`).
- It groups an individual user's movements into a single session based on time (e.g., if there is a gap of more than 90 minutes, it assumes a new session has started).
- The output is a list of transactions where each transaction is a sequence of locations visited by a user in one continuous timeframe.

### 3. `pattern_mining.py`
**Purpose**: Frequent Pattern Extraction
- It reads the processed `transactions.csv` and applies standard Data Mining algorithms: **Apriori** and **FP-Growth** (via the `mlxtend` library).
- It finds "frequent itemsets" (locations commonly visited in the same session) based on a minimum support threshold.
- It extracts "Association Rules" (e.g., If someone visits the Library, they have an X% confidence of visiting the Cafeteria next).

### 4. `graph_analysis.py`
**Purpose**: Network & Hotspot Analysis
- It reads the processed `transactions.csv` and builds a directed graph (using the `networkx` library) where nodes are locations and edges represent the transition of users from one location to another.
- It calculates graph metrics, most notably **PageRank**, to identify the central "hotspots" of the campus.
- It analyzes edge weights to find the "heaviest traffic paths" (bottlenecks) on campus.

### 5. `classification.py` (NEW)
**Purpose**: Supervised Learning (Classification)
- It transforms the raw log data into a supervised learning dataset where features are the user's `Current_Location`, `Hour`, and `DayOfWeek`, and the target is the `Next_Location`.
- It implements two classification algorithms from `scikit-learn`:
  - **Naïve Bayes (`CategoricalNB`)**: Uses Bayes' theorem and probability theory to calculate the most likely next destination.
  - **K-Nearest Neighbors (`KNeighborsClassifier`)**: A lazy learner that looks at the $K$ most historically similar movement instances and takes a majority vote to predict the next destination.
- It provides functions to train, evaluate, and use these models for real-time predictions.

### 6. `dashboard.py`
**Purpose**: User Interface & Visualization
- This is the main front-end application built using **Streamlit**.
- It acts as an interactive dashboard that ties all the other scripts together.
- Users can use the sidebar to trigger data generation and processing.
- It displays the results of the pattern mining (comparing the execution speed of Apriori vs FP-Growth) and lists the top movement rules.
- It visualizes the campus graph interactively (using `pyvis`), showing nodes sized by their PageRank (hotspot level) and edges thickened by traffic volume.
- **Classification Section**: It provides an interactive tool to test the Naïve Bayes and KNN models, allowing users to input a scenario and see the predicted next location.
