import numpy as np
import pandas as pd
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

# Adding Tenure Tiers
def categorize_tenure(years):
    if years <= 2:
        return 'New'
    elif years <= 5:
        return 'Established'
    else:
        return 'Long-Term'

# Apply the function to create the new column
dff['TenureTier'] = dff['Tenure'].apply(categorize_tenure)

dff.to_csv("Modified_European_Bank.csv", index=False)
