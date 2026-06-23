# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
from src.recommender_engine import OutdoorRecommender
import re

# Set up page configurations
st.set_page_config(
    page_title="AI Outdoor Gear Recommender",
    page_icon="🏔️",
    layout="wide"
)

# Cache the engine loading process so it only runs once at startup
@st.cache_resource
def load_engine():
    return OutdoorRecommender()

try:
    engine = load_engine()
except Exception as e:
    st.error(f"Failed to initialize the recommendation engine artifacts: {e}")
    st.stop()

# Header Design Element
st.title("🏔️ Smart Outdoor Gear Discovery Engine")
st.markdown("---")

# Setup Sidebar Context Selection Mode
app_mode = st.sidebar.selectbox(
    "Choose Discovery Interface",
    ["🧠 Conceptual Semantic Search", "🏷️ Item-to-Item Recommendations"]
)

st.sidebar.markdown("""
### How it works:
This intelligent backend converts raw textual profiles into dense **384-dimensional vector coordinates** utilizing an `all-MiniLM-L6-v2` Transformer, scoring recommendations in real-time via angular **Cosine Similarity**.
""")

# Utility function to render item grids with clean layout architecture
def display_product_cards(dataframe):
    if dataframe.empty:
        st.warning("No product entries to display.")
        return
        
    # Generate responsive layout columns
    columns = st.columns(len(dataframe))
    
    for idx, (col, (_, row)) in enumerate(zip(columns, dataframe.iterrows())):
        with col:
            with st.container(border=True):
                
                # Fetch catalog images safely
                img_url = row.get('image', '') if pd.notna(row.get('image', '')) else ''
                if img_url:
                    st.image(img_url, use_container_width=True)
                else:
                    st.image("https://placehold.co/200x200?text=No+Image", use_container_width=True)
                
                # Robust regex-based clean-up engine
                clean_name = str(row['name'])
                
                # 1. Clean brand prefix matches (e.g., "SIMOND - Climbing...")
                brand_prefix = f"{row['brand']} - "
                if clean_name.startswith(brand_prefix):
                    clean_name = clean_name[len(brand_prefix):]
                
                # 2. Use word-boundary regex to wipe out unwanted terms anywhere in the string
                # This matches the term cleanly even if it's right at the end of the line
                unwanted_patterns = r'\b(St|He|Mix|30mbar|50mbar|L/S|S/S)\.?\b'
                clean_name = re.sub(unwanted_patterns, '', clean_name, flags=re.IGNORECASE)
                
                # 3. Collapse extra whitespace and trim the edges
                clean_name = " ".join(clean_name.split()).strip()
                
                # Render UI Elements
                st.markdown(f"### **{row['brand']}**")
                st.markdown(f"*{clean_name}*")
                
                st.markdown(f"**Category:** `{row['category'].upper()}`")
                st.markdown(f"**Price:** <span style='color:#2e7d32; font-weight:bold;'>{row['price_clean']} EUR</span>", unsafe_allow_html=True)
                
                if row['rating_score'] > 0:
                    stars = "⭐" * int(round(row['rating_score']))
                    st.markdown(f"**Rating:** {stars} ({row['rating_score']} / 5)")
                else:
                    st.markdown("**Rating:** 🚫 No reviews yet")
                
                if row['discount_percentage'] > 0:
                    st.markdown(f"**Deal:** <span style='color:#c62828; font-weight:bold;'>-{row['discount_percentage']}% Off</span>", unsafe_allow_html=True)
                else:
                    st.markdown("**Deal:** Regular Price")
                
                if 'url' in row and pd.notna(row['url']):
                    st.link_button("View Shop Item ↗", row['url'], use_container_width=True)

# --- MODE 1: SEMANTIC SEARCH (UPDATED FOR HYBRID) ---
if app_mode == "🧠 Conceptual Semantic Search":
    st.header("🧠 Intelligent Semantic Context Search")
    st.markdown("Enter natural phrases describing what you need. The engine evaluates semantic concepts alongside ratings and discount depth.")
    
    search_query = st.text_input(
        "What kind of outdoor adventure are you planning?", 
        placeholder="e.g., highly insulated winter mountaineering jacket on discount"
    )
    
    num_results = st.slider("Max items to surface", min_value=3, max_value=12, value=4)
    
    if search_query:
        with st.spinner("Scanning dense vector universe and sorting ranking weights..."):
            # CALL THE NEW HYBRID FUNCTION HERE ⚡
            results = engine.hybrid_semantic_search(search_query, top_n=num_results)
            full_results = engine.df.loc[results.index]
            
            st.success(f"Discovered top match solutions for: '{search_query}'")
            display_product_cards(full_results)

# --- MODE 2: PRODUCT RECOMMENDATIONS (UPDATED FOR HYBRID) ---
else:
    st.header("🏷️ Cross-Product Similarity Discovery Engine")
    st.markdown("Select an existing catalog item to discover complementary gear calculated by text alignment, product ratings, and deep clearance value.")
    
    available_items = sorted(engine.df['name'].unique())
    selected_item = st.selectbox("Select a reference catalog item:", available_items)
    
    num_recs = st.slider("Complementary choices to look up", min_value=3, max_value=12, value=4)
    
    if selected_item:
        with st.spinner("Calculating hybrid affinity matrix scores..."):
            # CALL THE NEW HYBRID FUNCTION HERE ⚡
            recs = engine.get_hybrid_recommendations(selected_item, top_n=num_recs)
            full_recs = engine.df.loc[recs.index]
            
            st.subheader(f"Because you showed interest in: **{selected_item}**")
            display_product_cards(full_recs)