import networkx as nx
import pandas as pd

def build_graph_from_transactions(file_path='transactions.csv'):
    df = pd.read_csv(file_path)
    transactions = df['Transaction_Str'].apply(lambda x: x.split(',')).tolist()
    
    # Directed Graph
    G = nx.DiGraph()
    
    # Count edge frequencies
    edge_counts = {}
    for t in transactions:
        for i in range(len(t) - 1):
            source = t[i]
            target = t[i+1]
            if source == target:
                continue # Skip self loops if any
            edge = (source, target)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
            
    # Add edges with weight
    for (u, v), weight in edge_counts.items():
        G.add_edge(u, v, weight=weight)
        
    return G

def analyze_graph(G):
    # Calculate degree centrality (identifies hotspots)
    in_degree = nx.in_degree_centrality(G)
    out_degree = nx.out_degree_centrality(G)
    pagerank = nx.pagerank(G, weight='weight')
    
    # Combine metrics into a DataFrame for easy viewing
    nodes = list(G.nodes())
    metrics = []
    for node in nodes:
        metrics.append({
            'Location': node,
            'In-Degree': in_degree[node],
            'Out-Degree': out_degree[node],
            'PageRank': pagerank[node]
        })
        
    df_metrics = pd.DataFrame(metrics).sort_values(by='PageRank', ascending=False)
    
    # Identify bottleneck/heaviest traffic paths
    edges = list(G.edges(data=True))
    edges.sort(key=lambda x: x[2]['weight'], reverse=True)
    top_edges = [{'Source': u, 'Target': v, 'Traffic Volume': d['weight']} for u, v, d in edges[:10]]
    df_edges = pd.DataFrame(top_edges)
    
    return df_metrics, df_edges

if __name__ == '__main__':
    G = build_graph_from_transactions()
    df_metrics, df_edges = analyze_graph(G)
    
    print("--- Campus Hotspots (PageRank) ---")
    print(df_metrics.to_string(index=False))
    
    print("\n--- Heaviest Traffic Paths ---")
    print(df_edges.to_string(index=False))
