import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

from data_generator import generate_campus_data
from preprocessing import process_logs_to_transactions
from pattern_mining import run_pattern_mining, load_transactions
from graph_analysis import build_graph_from_transactions, analyze_graph
import os

st.set_page_config(page_title="Smart Campus Data Mining", layout="wide")

st.title("🏛️ Smart Campus Crowd Movement Analysis")
st.markdown("This dashboard demonstrates Data Mining techniques (Frequent Pattern Mining & Graph Analysis) applied to synthetic campus WiFi movement logs.")

# Sidebar controls
st.sidebar.header("Controls")
num_users = st.sidebar.slider("Number of Users", 100, 1000, 500, step=100)
min_support = st.sidebar.slider("Min Support (Pattern Mining)", 0.01, 0.20, 0.01, step=0.01)

if st.sidebar.button("1. Generate & Process Data"):
    with st.spinner("Generating raw logs..."):
        generate_campus_data(num_users=num_users, days=7)
    with st.spinner("Sessionizing into transactions..."):
        process_logs_to_transactions()
    st.sidebar.success("Data generated and processed!")

if os.path.exists("transactions.csv"):
    st.header("📊 1. Frequent Pattern Mining (Apriori vs FP-Growth)")
    
    transactions = load_transactions()
    st.write(f"Loaded **{len(transactions)}** movement sessions.")
    
    with st.spinner("Mining Patterns..."):
        results, rules = run_pattern_mining(transactions, min_support=min_support)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Apriori Algorithm")
        st.metric("Execution Time", f"{results['Apriori']['time']:.4f} s")
        st.metric("Itemsets Found", len(results['Apriori']['itemsets']))
        
    with col2:
        st.subheader("FP-Growth Algorithm")
        st.metric("Execution Time", f"{results['FP-Growth']['time']:.4f} s")
        st.metric("Itemsets Found", len(results['FP-Growth']['itemsets']))
        
    if results['Apriori']['time'] > 0 and results['FP-Growth']['time'] > 0:
        speedup = results['Apriori']['time'] / results['FP-Growth']['time']
        st.info(f"**FP-Growth** is roughly **{speedup:.2f}x faster** than Apriori on this dataset.")
        
    st.subheader("Top Movement Rules (Association Rules)")
    if not rules.empty:
        # Format rules for display
        display_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
        display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
        st.dataframe(display_rules.head(15))
    else:
        st.warning("No rules found with current min_support. Try lowering it.")
        
    st.header("🕸️ 2. Graph Analysis (Hotspots & Bottlenecks)")
    
    G = build_graph_from_transactions()
    df_metrics, df_edges = analyze_graph(G)
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Campus Hotspots (PageRank)")
        st.dataframe(df_metrics)
    with col4:
        st.subheader("Heaviest Traffic Paths (Edges)")
        st.dataframe(df_edges)
        
    st.subheader("Interactive Campus Network")
    # Pyvis Network
    net = Network(height='500px', width='100%', directed=True, bgcolor='#ffffff', font_color='black')
    
    # Add nodes (size based on PageRank)
    max_pr = df_metrics['PageRank'].max()
    for _, row in df_metrics.iterrows():
        size = (row['PageRank'] / max_pr) * 50 + 10
        net.add_node(row['Location'], label=row['Location'], size=size, title=f"PageRank: {row['PageRank']:.3f}")
        
    # Add edges
    if not df_edges.empty:
        max_weight = df_edges['Traffic Volume'].max()
        for _, row in df_edges.iterrows():
            # Edge width based on traffic
            width = (row['Traffic Volume'] / max_weight) * 10 + 1
            net.add_edge(row['Source'], row['Target'], value=width, title=f"Traffic: {row['Traffic Volume']}")
            
    net.save_graph("campus_graph.html")
    
    HtmlFile = open("campus_graph.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, height=520)

else:
    st.info("👈 Please click '1. Generate & Process Data' in the sidebar to begin.")
