import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# Data Ingestion & Validation
df = pd.read_csv("European_Bank.csv")
dff=df.copy()

null_counts = dff[['IsActiveMember', 'HasCrCard', 'NumOfProducts', 'Exited']].isnull().sum()
print("\n[1] Missing Value Count:")
print(null_counts)

for field in ['IsActiveMember', 'HasCrCard', 'Exited']:
    dff[field]=dff[field].astype(int)
    minimum = dff[field].min()
    maximum = dff[field].max()
    if minimum < 0 or maximum > 1:
        print(f"CRITICAL ERROR: {field} contains numbers other of 0 and 1!")
    else:
        print(f"{field} passed check for binary digits.")

# Engagement Classification
median_balance = dff['Balance'].median()
def classify_engagement(row, med_bal):
    # Active engaged customers
    if row['IsActiveMember'] == 1 and row['NumOfProducts'] >= 2:
        return 'Active Engaged'
    
    # Inactive disengaged customers
    elif row['IsActiveMember'] == 0 and row['NumOfProducts'] == 1:
        return 'Inactive Disengaged'
    
    # Active but low-product customers
    elif row['IsActiveMember'] == 1 and row['NumOfProducts'] == 1:
        return 'Active Low-Product'
    
    # Inactive high-balance customers
    elif row['IsActiveMember'] == 0 and row['Balance'] > med_bal:
        return 'Inactive High-Balance'
    
    # rows that don't perfectly fit the above
    else:
        return 'Standard'

# create your new column
dff['EngagementProfile'] = dff.apply(lambda r: classify_engagement(r, median_balance), axis=1)

# Product Utilization Analysis
# Churn rate by number of products
product_churn_analysis = dff.groupby('NumOfProducts')['Exited'].mean() * 100
print(product_churn_analysis)

# Single-product vs multi-product retention
dff['IsMultiProduct'] = dff['NumOfProducts'] > 1
multi_product_analysis = dff.groupby('IsMultiProduct')['Exited'].mean() * 100
print(multi_product_analysis)

# Product depth vs churn relationship
dff['ProductDepthScore'] = (dff['NumOfProducts'] / 4) * 100
portfolio_depth_churn = dff.groupby('ProductDepthScore')['Exited'].mean() * 100
print(portfolio_depth_churn)

# Financial Commitment vs Engagement Analysis
# Balance vs. Activity Cross-Analysis
dff['BalanceTier'] = pd.qcut(dff['Balance'].rank(method='first'), q=3, labels=['Low', 'Medium', 'High'])
balance_activity_cross = pd.crosstab(
    index=dff['BalanceTier'], 
    columns=dff['IsActiveMember'], 
    values=dff['Exited'], 
    aggfunc='mean'
) * 100
balance_activity_cross.columns = ['Inactive Churn Rate (%)', 'Active Churn Rate (%)']
print(balance_activity_cross)

# Salary–Balance Mismatch Detection
dff['BalanceToSalaryRatio'] = dff['Balance'] / (dff['EstimatedSalary'] + 1)
salary_threshold = dff['EstimatedSalary'].quantile(0.70)
ratio_threshold = dff['BalanceToSalaryRatio'].quantile(0.30)
dff['FinancialMismatch'] = (dff['EstimatedSalary'] >= salary_threshold) & (dff['BalanceToSalaryRatio'] <= ratio_threshold) 

#Identification of “At-Risk Premium Customers”
premium_balance_threshold = dff['Balance'].quantile(0.75)
dff['AtRiskPremium'] = (dff['Balance'] >= premium_balance_threshold) & (dff['IsActiveMember'] == 0) & (dff['Exited'] == 0)

# Retention Strength Assessment
# Define "Sticky Customer" Profiles
dff['IsStickyCustomer'] = (
    (dff['IsActiveMember'] == 1) & 
    (dff['NumOfProducts'] == 2) & 
    (dff['HasCrCard'] == 1) & 
    (dff['Tenure'] > 5)
)
sticky_profile_churn = dff.groupby('IsStickyCustomer')['Exited'].mean() * 100
print(sticky_profile_churn)

# Measure Churn Stability Across Engagement Tiers
engagement_tier_stability = dff.groupby('EngagementProfile')['Exited'].agg(
    Total_Customers='count',
    Churn_Rate_Pct=lambda x: x.mean() * 100
).reset_index()
print(engagement_tier_stability)

# Identify Engagement Thresholds Linked to Retention
dff["RSI"] = (
    (dff["IsActiveMember"] * 3)
    + (np.minimum(dff["NumOfProducts"], 3) * 1)
    + (dff["HasCrCard"] * 2)
    + (np.where(dff["Tenure"] > 5, 2, 1))
)
rsi_threshold_analysis = (
    dff.groupby("RSI")["Exited"]
    .agg(Volume="count", Churn_Rate_Pct=lambda x: x.mean() * 100)
    .reset_index()
)
print(rsi_threshold_analysis)

dff.to_csv("Modified_European_Bank.csv", index=False)

# Set up page configurations
st.set_page_config(page_title="Bank Retention Analytics Dashboard", layout="wide", page_icon="📊")

# DATA LOADING & INITIALIZATION
@st.cache_data
def load_data():
    # Reads the file you generated and saved at the end of Phase 1
    return pd.read_csv("Modified_European_Bank.csv")
try:
    dff = load_data()
except FileNotFoundError:
    st.error("❌ 'Modified_European_Bank.csv' not found! Please run your backend script first to generate the dataset.")
    st.stop()

# Title and App Header
st.title("📊 Customer Retention & Engagement Dashboard")
st.markdown("An advanced behavioral analytics panel tracking financial commitment, product utility, and churn risk.")
st.markdown("---")

# USER CAPABILITIES (SIDEBAR WIDGETS)
st.sidebar.header("🛠️ Control Panel & Filters")

# 1. Engagement Filters
all_profiles = dff['EngagementProfile'].unique().tolist()
selected_profiles = st.sidebar.multiselect(
    "Select Engagement Profiles:", 
    options=all_profiles, 
    default=all_profiles
)

# 2. Product Count Slider
min_prod, max_prod = int(dff['NumOfProducts'].min()), int(dff['NumOfProducts'].max())
selected_products = st.sidebar.slider(
    "Filter by Product Count:", 
    min_value=min_prod, max_value=max_prod, 
    value=(min_prod, max_prod)
)

# 3. Financial Thresholds
st.sidebar.markdown("### 💰 Financial Tiers")
max_balance = float(dff['Balance'].max())
balance_threshold = st.sidebar.number_input(
    "Minimum Balance Filter ($):", 
    min_value=0.0, max_value=max_balance, value=0.0, step=5000.0
)

# Apply Sidebar Filters to Data
filtered_dff = dff[
    (dff['EngagementProfile'].isin(selected_profiles)) &
    (dff['NumOfProducts'].between(selected_products[0], selected_products[1])) &
    (dff['Balance'] >= balance_threshold)
]

# Quick Metric Row at the Top
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Filtered Portfolio Volume", f"{filtered_dff.shape[0]:,} Users")
with m2:
    current_churn = filtered_dff['Exited'].mean() * 100 if not filtered_dff.empty else 0
    st.metric("Portfolio Churn Rate", f"{current_churn:.2f}%")
with m3:
    st.metric("Total Vault Balance Assured", f"${filtered_dff['Balance'].sum():,.2f}")

st.markdown("---")

# =========================================================
# CORE MODULE 1: ENGAGEMENT VS CHURN OVERVIEW
# =========================================================
st.header("1. Engagement vs. Churn Overview")
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Churn Rate by Behavioral Profile")
    # Grouping data based on your engineered engagement profiles
    engagement_analysis = filtered_dff.groupby('EngagementProfile')['Exited'].mean().reset_index()
    engagement_analysis['Churn Rate (%)'] = engagement_analysis['Exited'] * 100
    
    fig_eng = px.bar(
        engagement_analysis, x='EngagementProfile', y='Churn Rate (%)',
        color='EngagementProfile', text_auto='.1f',
        title="Attrition Probability Across Engagement Tiers"
    )
    st.plotly_chart(fig_eng, use_container_width=True)

with col2:
    st.subheader("Profile Portfolio Distribution")
    fig_pie = px.pie(
        filtered_dff, names='EngagementProfile', 
        title="Share of Total Bank Accounts by Segment",
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# =========================================================
# CORE MODULE 2: PRODUCT UTILIZATION IMPACT ANALYSIS
# =========================================================
st.header("2. Product Utilization Impact Analysis")
col3, col4 = st.columns([1, 1])

with col3:
    st.subheader("Product Depth vs. Churn Relationship")
    # Visualizing the raw counts check or depth scores
    prod_churn = filtered_dff.groupby('NumOfProducts')['Exited'].mean().reset_index()
    prod_churn['Churn Rate (%)'] = prod_churn['Exited'] * 100
    
    fig_prod = px.line(
        prod_churn, x='NumOfProducts', y='Churn Rate (%)', 
        markers=True, title="The Product Depth Risk Curve (U-Shape)"
    )
    st.plotly_chart(fig_prod, use_container_width=True)

with col4:
    st.subheader("Retention Breakdown: Single vs. Multi-Product")
    if 'Is_Multi_Product' in filtered_dff.columns:
        multi_churn = filtered_dff.groupby('Is_Multi_Product')['Exited'].mean().reset_index()
        multi_churn['Churn Rate (%)'] = multi_churn['Exited'] * 100
        multi_churn['Portfolio Group'] = multi_churn['Is_Multi_Product'].map({True: 'Multi-Product (2+)', False: 'Single Product (1)'})
        
        fig_multi = px.bar(
            multi_churn, x='Portfolio Group', y='Churn Rate (%)', 
            color='Portfolio Group', text_auto='.1f', title="Retention Anchoring Analysis"
        )
        st.plotly_chart(fig_multi, use_container_width=True)
    else:
        st.info("Run your complete backend to display Single vs Multi-Product comparisons.")

st.markdown("---")

# =========================================================
# CORE MODULE 3: HIGH-VALUE DISENGAGED CUSTOMER DETECTOR
# =========================================================
st.header("3. High-Value Disengaged Customer Detector")
tab1, tab2 = st.tabs(["⚠️ Salary-Balance Mismatches", "🚨 Targetable At-Risk Premium Users"])

with tab1:
    st.subheader("Identified Structural Mismatches")
    st.markdown("High salary earners keeping exceptionally low balance depth relative to income.")
    if 'Financial_Mismatch' in filtered_dff.columns:
        mismatch_view = filtered_dff[filtered_dff['Financial_Mismatch'] == True][['CustomerId', 'CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'Balance_To_Salary_Ratio']]
        st.dataframe(mismatch_view.head(100), use_container_width=True)
    else:
        st.info("Mismatch flags column not detected in source data layout.")

with tab2:
    st.subheader("Premium Value 'Ghost Accounts' At Imminent Risk")
    st.markdown("Top 25% asset holders who are fully inactive but have not officially initiated account closing sequence yet.")
    if 'At_Risk_Premium' in filtered_dff.columns:
        premium_view = filtered_dff[filtered_dff['At_Risk_Premium'] == True][['CustomerId', 'CreditScore', 'Age', 'Geography', 'Gender', 'Balance', 'NumOfProducts']]
        st.dataframe(premium_view.head(100), use_container_width=True)
        
        # Download button capability for marketing operations
        csv_data = premium_view.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export High-Priority Retention Target List",
            data=csv_data,
            file_name="high_priority_retention_targets.csv",
            mime="text/csv"
        )
    else:
        st.info("Premium target flags column not detected in source data layout.")

st.markdown("---")

# =========================================================
# CORE MODULE 4: RETENTION STRENGTH SCORING PANELS
# =========================================================
st.header("4. Retention Strength Scoring Panels (RSI)")
col5, col6 = st.columns([1, 1])

with col5:
    st.subheader("RSI Level Churn Distributions")
    if 'RSI' in filtered_dff.columns:
        rsi_analysis = filtered_dff.groupby('RSI')['Exited'].mean().reset_index()
        rsi_analysis['Churn Rate (%)'] = rsi_analysis['Exited'] * 100
        
        fig_rsi = px.bar(
            rsi_analysis, x='RSI', y='Churn Rate (%)',
            labels={'RSI': 'Relationship Strength Index (1-9)'},
            text_auto='.1f', title="Attrition Mitigation Over RSI Levels"
        )
        st.plotly_chart(fig_rsi, use_container_width=True)
    else:
        st.info("RSI score array sequence is not loaded.")

with col6:
    st.subheader("Strategic Threshold Invariant Metrics")
    st.markdown("**Analytical Baseline Interpretations:**")
    st.write("• **Volatile Zone (RSI 1 - 4):** Customer accounts hold extreme velocity profile characteristics. Heavy structural friction.")
    st.write("• **Defensive Anchor Point (RSI 5):** The critical inflection target threshold line where attrition structural margins plummet below portfolio benchmark averages.")
    st.write("• **Elite Stability Anchor (RSI 6+):** Complete portfolio integration. Safe, structural asset loyalty markers achieved.")