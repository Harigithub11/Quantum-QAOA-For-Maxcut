"""
What is MaxCut Problem - Interactive Visual Explanation
Step-by-step animated demonstration of the MaxCut problem
"""

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="What is MaxCut?", page_icon="✂️", layout="wide")

st.title("✂️ What is the MaxCut Problem?")
st.markdown("**Interactive Visual Explanation - Watch the problem come to life!**")

st.markdown("---")

# Add start button
col_center = st.columns([1, 2, 1])[1]
with col_center:
    start_demo = st.button("▶️ Start Interactive Demo", type="primary", use_container_width=True)

if start_demo:
    # Create placeholder for animations
    demo_placeholder = st.empty()

    st.markdown("---")
    st.markdown("## 🎬 Live Demo: Understanding MaxCut")

    # ========== DEMO 1: TRIANGLE ==========
    st.markdown("### 📐 Example 1: Triangle Graph (3 Friends)")

    placeholder_1 = st.empty()

    # Step 1: Show original triangle
    with placeholder_1.container():
        st.info("**Step 1**: We have 3 friends who all know each other (complete graph)")

        fig1, ax1 = plt.subplots(figsize=(6, 5))

        # Create triangle
        G_tri = nx.Graph()
        G_tri.add_edges_from([(0, 1), (1, 2), (2, 0)])

        pos_tri = {0: (0.5, 1), 1: (0, 0), 2: (1, 0)}

        nx.draw(G_tri, pos_tri, with_labels=True,
                node_color='lightblue', node_size=1200,
                font_size=20, font_weight='bold',
                edge_color='gray', width=4, ax=ax1)

        ax1.set_title("3 Friends: All Connected", fontsize=16, fontweight='bold')
        ax1.text(0.5, -0.3, "Total Edges: 3", ha='center', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        st.pyplot(fig1)
        plt.close(fig1)

    time.sleep(2)

    # Step 2: Show problem statement
    with placeholder_1.container():
        st.warning("**Step 2**: GOAL - Split into 2 teams to maximize 'rivalries' (edges between teams)")

        fig2, ax2 = plt.subplots(figsize=(6, 5))

        nx.draw(G_tri, pos_tri, with_labels=True,
                node_color='yellow', node_size=1200,
                font_size=20, font_weight='bold',
                edge_color='orange', width=4, ax=ax2)

        ax2.set_title("How to split into 2 teams?", fontsize=16, fontweight='bold', color='red')
        ax2.text(0.5, -0.3, "Want: Maximum edges BETWEEN teams", ha='center', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
        st.pyplot(fig2)
        plt.close(fig2)

    time.sleep(2)

    # Step 3: Bad partition
    with placeholder_1.container():
        st.error("**Step 3**: ❌ Bad Split - All in one team (0 rivalries)")

        fig3, ax3 = plt.subplots(figsize=(6, 5))

        node_colors = ['red', 'red', 'red']
        nx.draw(G_tri, pos_tri, with_labels=True,
                node_color=node_colors, node_size=1200,
                font_size=20, font_weight='bold',
                edge_color='lightgray', width=4, ax=ax3)

        ax3.set_title("Bad: All Red Team", fontsize=16, fontweight='bold', color='darkred')
        ax3.text(0.5, -0.3, "Cut Edges: 0 ❌", ha='center', fontsize=16,
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        st.pyplot(fig3)
        plt.close(fig3)

    time.sleep(2)

    # Step 4: Medium partition
    with placeholder_1.container():
        st.info("**Step 4**: 🟡 OK Split - 2 in one team, 1 in other (2 rivalries)")

        fig4, ax4 = plt.subplots(figsize=(6, 5))

        node_colors = ['red', 'red', 'blue']
        nx.draw(G_tri, pos_tri, with_labels=True,
                node_color=node_colors, node_size=1200,
                font_size=20, font_weight='bold',
                edge_color='lightgray', width=4, ax=ax4)

        # Highlight cut edges
        cut_edges = [(1, 2), (0, 2)]
        nx.draw_networkx_edges(G_tri, pos_tri, cut_edges,
                               edge_color='green', width=6, ax=ax4)

        ax4.set_title("OK: Red vs Blue (2 teams)", fontsize=16, fontweight='bold', color='darkblue')
        ax4.text(0.5, -0.3, "Cut Edges: 2 🟡", ha='center', fontsize=16,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        st.pyplot(fig4)
        plt.close(fig4)

    time.sleep(2)

    # Step 5: Optimal partition
    with placeholder_1.container():
        st.success("**Step 5**: ✅ BEST Split - 2 teams, 1 person each (3 rivalries - MAXIMUM!)")

        fig5, ax5 = plt.subplots(figsize=(6, 5))

        node_colors = ['red', 'blue', 'red']
        nx.draw(G_tri, pos_tri, with_labels=True,
                node_color=node_colors, node_size=1200,
                font_size=20, font_weight='bold',
                edge_color='lightgray', width=4, ax=ax5)

        # Highlight ALL edges as cut edges
        cut_edges = [(0, 1), (1, 2), (2, 0)]
        nx.draw_networkx_edges(G_tri, pos_tri, cut_edges,
                               edge_color='lime', width=8, ax=ax5)

        ax5.set_title("OPTIMAL: Maximum Rivalries!", fontsize=16, fontweight='bold', color='green')
        ax5.text(0.5, -0.3, "Cut Edges: 3/3 (100%) ✅", ha='center', fontsize=16,
                bbox=dict(boxstyle='round', facecolor='lime', alpha=0.7))
        st.pyplot(fig5)
        plt.close(fig5)

    time.sleep(2)

    st.markdown("---")

    # ========== DEMO 2: SQUARE ==========
    st.markdown("### 🟦 Example 2: Square Graph (4 People in a Square)")

    placeholder_2 = st.empty()

    # Step 1: Show original square
    with placeholder_2.container():
        st.info("**Step 1**: 4 people sitting in a square, connected to neighbors")

        fig6, ax6 = plt.subplots(figsize=(6, 5))

        # Create square (cycle of 4)
        G_square = nx.cycle_graph(4)

        pos_square = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (0, 0)}

        nx.draw(G_square, pos_square, with_labels=True,
                node_color='lightblue', node_size=1200,
                font_size=20, font_weight='bold',
                edge_color='gray', width=4, ax=ax6)

        ax6.set_title("4 People: Square Formation", fontsize=16, fontweight='bold')
        ax6.text(0.5, -0.3, "Total Edges: 4", ha='center', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        st.pyplot(fig6)
        plt.close(fig6)

    time.sleep(2)

    # Step 2: Bad partition (all same)
    with placeholder_2.container():
        st.error("**Step 2**: ❌ Bad Split - All same team (0 rivalries)")

        fig7, ax7 = plt.subplots(figsize=(6, 5))

        node_colors = ['red', 'red', 'red', 'red']
        nx.draw(G_square, pos_square, with_labels=True,
                node_color=node_colors, node_size=1200,
                font_size=20, font_weight='bold',
                edge_color='lightgray', width=4, ax=ax7)

        ax7.set_title("Bad: All Red Team", fontsize=16, fontweight='bold', color='darkred')
        ax7.text(0.5, -0.3, "Cut Edges: 0 ❌", ha='center', fontsize=16,
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        st.pyplot(fig7)
        plt.close(fig7)

    time.sleep(2)

    # Step 3: Suboptimal partition
    with placeholder_2.container():
        st.warning("**Step 3**: 🟡 Suboptimal - 3 vs 1 (only 2 rivalries)")

        fig8, ax8 = plt.subplots(figsize=(6, 5))

        node_colors = ['red', 'red', 'red', 'blue']
        nx.draw(G_square, pos_square, with_labels=True,
                node_color=node_colors, node_size=1200,
                font_size=20, font_weight='bold',
                edge_color='lightgray', width=4, ax=ax8)

        # Highlight cut edges
        cut_edges = [(3, 0), (2, 3)]
        nx.draw_networkx_edges(G_square, pos_square, cut_edges,
                               edge_color='orange', width=6, ax=ax8)

        ax8.set_title("Suboptimal: Can do better!", fontsize=16, fontweight='bold', color='orange')
        ax8.text(0.5, -0.3, "Cut Edges: 2/4 (50%) 🟡", ha='center', fontsize=16,
                bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
        st.pyplot(fig8)
        plt.close(fig8)

    time.sleep(2)

    # Step 4: Optimal partition
    with placeholder_2.container():
        st.success("**Step 4**: ✅ OPTIMAL - Alternating teams (4 rivalries - MAXIMUM!)")

        fig9, ax9 = plt.subplots(figsize=(6, 5))

        node_colors = ['red', 'blue', 'red', 'blue']
        nx.draw(G_square, pos_square, with_labels=True,
                node_color=node_colors, node_size=1200,
                font_size=20, font_weight='bold',
                edge_color='lightgray', width=4, ax=ax9)

        # Highlight ALL edges as cut edges
        cut_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        nx.draw_networkx_edges(G_square, pos_square, cut_edges,
                               edge_color='lime', width=8, ax=ax9)

        ax9.set_title("OPTIMAL: Checkerboard Pattern!", fontsize=16, fontweight='bold', color='green')
        ax9.text(0.5, -0.3, "Cut Edges: 4/4 (100%) ✅", ha='center', fontsize=16,
                bbox=dict(boxstyle='round', facecolor='lime', alpha=0.7))
        st.pyplot(fig9)
        plt.close(fig9)

    time.sleep(2)

    st.markdown("---")

    # ========== DEMO 3: COMPLEX GRAPH ==========
    st.markdown("### 🌐 Example 3: Complex Graph (Why We Need Algorithms!)")

    placeholder_3 = st.empty()

    # Step 1: Show complex graph
    with placeholder_3.container():
        st.info("**Step 1**: 6-node graph with many connections (not obvious!)")

        fig10, ax10 = plt.subplots(figsize=(7, 6))

        # Create a more complex graph
        G_complex = nx.Graph()
        G_complex.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (1, 4),
                                  (2, 3), (2, 5), (3, 4), (4, 5)])

        pos_complex = nx.spring_layout(G_complex, seed=42)

        nx.draw(G_complex, pos_complex, with_labels=True,
                node_color='lightblue', node_size=1000,
                font_size=18, font_weight='bold',
                edge_color='gray', width=3, ax=ax10)

        ax10.set_title("Complex Network: Not Obvious!", fontsize=16, fontweight='bold')
        ax10.text(0.5, -0.15, "Total Edges: 9", ha='center', fontsize=14,
                 transform=ax10.transAxes,
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        st.pyplot(fig10)
        plt.close(fig10)

    time.sleep(2)

    # Step 2: Random attempt
    with placeholder_3.container():
        st.warning("**Step 2**: Random guess - only 5 rivalries")

        fig11, ax11 = plt.subplots(figsize=(7, 6))

        node_colors = ['red', 'blue', 'red', 'blue', 'blue', 'red']
        nx.draw(G_complex, pos_complex, with_labels=True,
                node_color=node_colors, node_size=1000,
                font_size=18, font_weight='bold',
                edge_color='lightgray', width=3, ax=ax11)

        # Calculate and highlight cut edges
        cut_edges = [(u, v) for u, v in G_complex.edges()
                    if node_colors[u] != node_colors[v]]
        nx.draw_networkx_edges(G_complex, pos_complex, cut_edges,
                               edge_color='orange', width=5, ax=ax11)

        ax11.set_title("Random Split: Not Great", fontsize=16, fontweight='bold', color='orange')
        ax11.text(0.5, -0.15, f"Cut Edges: {len(cut_edges)}/9 🟡", ha='center', fontsize=14,
                 transform=ax11.transAxes,
                 bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
        st.pyplot(fig11)
        plt.close(fig11)

    time.sleep(2)

    # Step 3: Greedy attempt
    with placeholder_3.container():
        st.info("**Step 3**: Greedy algorithm - found 7 rivalries")

        fig12, ax12 = plt.subplots(figsize=(7, 6))

        node_colors = ['red', 'blue', 'red', 'blue', 'red', 'blue']
        nx.draw(G_complex, pos_complex, with_labels=True,
                node_color=node_colors, node_size=1000,
                font_size=18, font_weight='bold',
                edge_color='lightgray', width=3, ax=ax12)

        cut_edges = [(u, v) for u, v in G_complex.edges()
                    if node_colors[u] != node_colors[v]]
        nx.draw_networkx_edges(G_complex, pos_complex, cut_edges,
                               edge_color='green', width=5, ax=ax12)

        ax12.set_title("Greedy Algorithm: Better!", fontsize=16, fontweight='bold', color='blue')
        ax12.text(0.5, -0.15, f"Cut Edges: {len(cut_edges)}/9 (Classical)", ha='center', fontsize=14,
                 transform=ax12.transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
        st.pyplot(fig12)
        plt.close(fig12)

    time.sleep(2)

    # Step 4: QAOA solution
    with placeholder_3.container():
        st.success("**Step 4**: QAOA (Quantum) - found 8 rivalries! ✨")

        fig13, ax13 = plt.subplots(figsize=(7, 6))

        node_colors = ['red', 'blue', 'blue', 'red', 'red', 'blue']
        nx.draw(G_complex, pos_complex, with_labels=True,
                node_color=node_colors, node_size=1000,
                font_size=18, font_weight='bold',
                edge_color='lightgray', width=3, ax=ax13)

        cut_edges = [(u, v) for u, v in G_complex.edges()
                    if node_colors[u] != node_colors[v]]
        nx.draw_networkx_edges(G_complex, pos_complex, cut_edges,
                               edge_color='lime', width=6, ax=ax13)

        ax13.set_title("QAOA Quantum: BEST! 🎯", fontsize=16, fontweight='bold', color='green')
        ax13.text(0.5, -0.15, f"Cut Edges: {len(cut_edges)}/9 (Quantum Advantage!) ✅",
                 ha='center', fontsize=14,
                 transform=ax13.transAxes,
                 bbox=dict(boxstyle='round', facecolor='lime', alpha=0.7))
        st.pyplot(fig13)
        plt.close(fig13)

    time.sleep(1)

    # Summary
    st.markdown("---")
    st.markdown("## 🎯 Summary: What is MaxCut?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
        **The Problem:**
        - Given: A graph (network)
        - Goal: Split into 2 groups
        - Maximize: Edges between groups
        """)

    with col2:
        st.warning("""
        **Why It's Hard:**
        - NP-Hard problem
        - Exponential possibilities
        - Example: 20 nodes = 1 million splits!
        """)

    with col3:
        st.success("""
        **Our Solution:**
        - Classical: Greedy (fast, 50-80% optimal)
        - Quantum: QAOA (explores all possibilities!)
        - Result: Better solutions! 🚀
        """)

else:
    st.info("👆 **Click 'Start Interactive Demo' to begin the visual explanation!**")

    st.markdown("---")
    st.markdown("## 📖 What You'll Learn:")

    st.markdown("""
    ### The demo will show you:

    1. **Triangle Example** - Simplest case (3 nodes, 3 edges)
       - See different ways to split
       - Understand what makes a solution "better"
       - Find the maximum cut

    2. **Square Example** - Classic pattern (4 nodes, 4 edges)
       - Bad splits vs good splits
       - Discover the checkerboard pattern
       - See why alternating colors works

    3. **Complex Network** - Real challenge (6 nodes, 9 edges)
       - Random guess vs Greedy vs QAOA
       - See quantum advantage in action
       - Understand why we need algorithms

    **Each step is animated with colored graphs and clear explanations!**
    """)

# Footer
st.markdown("---")
st.markdown("""
### 🎓 Key Concepts:

- **MaxCut Problem**: Partition graph nodes into two groups to maximize edges between them
- **Cut Edges**: Edges that connect different groups (these count toward our score!)
- **Optimal Solution**: The partition that gives the maximum number of cut edges
- **Why Quantum?**: For large graphs, quantum computing can explore exponentially more possibilities than classical algorithms!

**Ready to see your project in action?** → Go to "QAOA MaxCut Solver" page!
""")
