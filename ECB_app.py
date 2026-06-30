import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Set up page configurations
st.set_page_config(page_title="Bank Retention Analytics Dashboard", layout="wide", page_icon="📊")

# DATA LOADING & INITIALIZATION
@st.cache_data
def load_data():
    # Reads your custom file path name verbatim
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

# MANDATORY KEY PERFORMANCE INDICATORS
st.header("🔑 Key Performance Indicators (KPIs)")
m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    # 1. Engagement Retention Ratio: Active Churn vs Inactive Churn comparison
    active_churn = filtered_dff[filtered_dff['IsActiveMember'] == 1]['Exited'].mean() * 100 if not filtered_dff[filtered_dff['IsActiveMember'] == 1].empty else 0
    inactive_churn = filtered_dff[filtered_dff['IsActiveMember'] == 0]['Exited'].mean() * 100 if not filtered_dff[filtered_dff['IsActiveMember'] == 0].empty else 0
    st.metric(
        label="Engagement Retention Ratio", 
        value=f"{active_churn:.1f}% / {inactive_churn:.1f}%",
        help="Active Member Churn Rate vs Inactive Member Churn Rate"
    )

with m2:
    # 2. Product Depth Index: Products used vs loyalty (Average products owned)
    avg_products = filtered_dff['NumOfProducts'].mean() if not filtered_dff.empty else 0
    st.metric(
        label="Product Depth Index", 
        value=f"{avg_products:.2f}",
        help="Average number of products held per customer account"
    )

with m3:
    # 3. High-Balance Disengagement Rate: Premium churn risk
    premium_mask = filtered_dff['AtRiskPremium'] == True if 'AtRiskPremium' in filtered_dff.columns else filtered_dff['Balance'] > 0
    high_bal_disengaged = filtered_dff[premium_mask].shape[0]
    st.metric(
        label="High-Balance Disengagement", 
        value=f"{high_bal_disengaged:,} Accounts",
        help="Count of premium asset accounts showing zero active engagement flags"
    )

with m4:
    # 4. Credit Card Stickiness Score: Card ownership retention impact
    cc_churn = filtered_dff[filtered_dff['HasCrCard'] == 1]['Exited'].mean() * 100 if not filtered_dff[filtered_dff['HasCrCard'] == 1].empty else 0
    st.metric(
        label="Credit Card Stickiness", 
        value=f"{cc_churn:.1f}% Churn",
        help="Observed attrition rate among customers holding active credit accounts"
    )

with m5:
    # 5. Relationship Strength Index: Combined engagement & product score
    avg_rsi = filtered_dff['RSI'].mean() if 'RSI' in filtered_dff.columns else 0
    st.metric(
        label="Relationship Strength (RSI)", 
        value=f"{avg_rsi:.2f} / 9.0",
        help="Overall portfolio core metric average based on calculated RSI weights"
    )

st.markdown("---")

# ENGAGEMENT VS CHURN OVERVIEW
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

# PRODUCT UTILIZATION IMPACT ANALYSIS
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
    if 'IsMultiProduct' in filtered_dff.columns:
        multi_churn = filtered_dff.groupby('IsMultiProduct')['Exited'].mean().reset_index()
        multi_churn['Churn Rate (%)'] = multi_churn['Exited'] * 100
        multi_churn['Portfolio Group'] = multi_churn['IsMultiProduct'].map({True: 'Multi-Product (2+)', False: 'Single Product (1)'})
        
        fig_multi = px.bar(
            multi_churn, x='Portfolio Group', y='Churn Rate (%)', 
            color='Portfolio Group', text_auto='.1f', title="Retention Anchoring Analysis"
        )
        st.plotly_chart(fig_multi, use_container_width=True)
    else:
        st.info("Run your complete backend to display Single vs Multi-Product comparisons.")

st.markdown("---")

# HIGH-VALUE DISENGAGED CUSTOMER DETECTOR
st.header("3. High-Value Disengaged Customer Detector")
tab1, tab2 = st.tabs(["⚠️ Salary-Balance Mismatches", "🚨 Targetable At-Risk Premium Users"])

with tab1:
    st.subheader("Identified Structural Mismatches")
    st.markdown("High salary earners keeping exceptionally low balance depth relative to income.")
    if 'FinancialMismatch' in filtered_dff.columns:
        mismatch_view = filtered_dff[filtered_dff['FinancialMismatch'] == True][['CustomerId', 'CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'BalanceToSalaryRatio']]
        st.dataframe(mismatch_view.head(100), use_container_width=True)
    else:
        st.info("Mismatch flags column not detected in source data layout.")

with tab2:
    st.subheader("Premium Value 'Ghost Accounts' At Imminent Risk")
    st.markdown("Top 25% asset holders who are fully inactive but have not officially initiated account closing sequence yet.")
    if 'AtRiskPremium' in filtered_dff.columns:
        premium_view = filtered_dff[filtered_dff['AtRiskPremium'] == True][['CustomerId', 'CreditScore', 'Age', 'Geography', 'Gender', 'Balance', 'NumOfProducts']]
        st.dataframe(premium_view.head(100), use_container_width=True)
        
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

# RETENTION STRENGTH SCORING PANELS
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