import os
import pandas as pd

INPUT_FILE = "data/products.csv"
OUTPUT_FILE = "data/products_engineered.csv"

def generate_product_profiles():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found. Please verify your data directory setup.")
        return
        
    print("⏳ Loading cleaned product catalog data...")
    df = pd.read_csv(INPUT_FILE)
    
    print(f"📋 Loaded {len(df)} items. Beginning text compilation pipeline...")
    
    # Fill any latent empty strings just in case
    df['name'] = df['name'].fillna('Unknown Product')
    df['brand'] = df['brand'].fillna('Unknown Brand')
    df['category'] = df['category'].fillna('General Gear')
    
    # Compile text features into a single rich semantic block
    def build_metadata_string(row):
        profile = (
            f"Brand: {row['brand']}. "
            f"Product Name: {row['name']}. "
            f"Category: {row['category']}. "
            f"Priced at {row['price_clean']} EUR with a {row['discount_percentage']}% discount. "
        )
        
        # Factor user ratings into text weighting expressions
        if row['rating_score'] > 0:
            profile += f"This highly rated gear scored {row['rating_score']} out of 5 based on {int(row['review_count'])} customer reviews."
        else:
            profile += "New release product with no customer feedback evaluations available yet."
            
        return profile

    # Generate the engineered features
    df['metadata_profile'] = df.apply(build_metadata_string, axis=1)
    
    # Save the updated data matrix
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"🎉 Success! Metadata profiles engineered and stored directly in {OUTPUT_FILE}")
    
    # Preview a sample profile text block
    print("\n--- 🔍 Engineered Profile Text Preview (Item Index 0) ---")
    print(df['metadata_profile'].iloc[0])
    print("----------------------------------------------------------")

if __name__ == "__main__":
    generate_product_profiles()