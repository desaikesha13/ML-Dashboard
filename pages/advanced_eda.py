import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import eda_utils
from utils.eda_utils import AdvancedEDA

def show_advanced_eda():
    st.title("🔬 Advanced Exploratory Data Analysis")
    st.markdown("---")
    
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found! Please upload data first from the main page.")
        return

    data = st.session_state['df']  # ✅ This is correct
    
    # Initialize Advanced EDA
    advanced_eda = AdvancedEDA(data)
    
    # Sidebar for analysis selection
    st.sidebar.title("🎯 Advanced Analysis Options")
    analysis_options = st.sidebar.multiselect(
        "Select analysis to perform:",
        [
            "Outlier Detection",
            "Feature Importance",
            "Dimensionality Reduction",
            "Statistical Tests",
            "Data Quality Report"
        ],
        default=["Data Quality Report"]
    )
    
    # Display data info
    with st.expander("📊 Dataset Information", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{data.shape[0]:,}")
        with col2:
            st.metric("Total Columns", f"{data.shape[1]:,}")
        with col3:
            st.metric("Missing Values", f"{data.isnull().sum().sum():,}")
        with col4:
            st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        st.dataframe(data.head(), use_container_width=True)
    
    # Execute selected analyses
    if "Data Quality Report" in analysis_options:
        eda_utils.data_quality_report(st.session_state.df)
        st.markdown("---")
    
    if "Outlier Detection" in analysis_options:
        advanced_eda.outlier_detection()
        st.markdown("---")
    
    if "Feature Importance" in analysis_options:
        advanced_eda.feature_importance_analysis()
        st.markdown("---")
    
    if "Dimensionality Reduction" in analysis_options:
        advanced_eda.dimensionality_reduction()
        st.markdown("---")
    
    if "Statistical Tests" in analysis_options:
        advanced_eda.statistical_tests()
        st.markdown("---")
    
    # Additional insights section
    st.subheader("💡 Key Insights & Recommendations")
    
    insights = generate_insights(data)
    for insight in insights:
        st.write(f"• {insight}")

def generate_insights(data):
    """Generate automated insights about the data"""
    insights = []
    
    # 1. Missing values
    missing_vals = data.isnull().sum().sum()
    if missing_vals > 0:
        insights.append(f"Detected {missing_vals} missing values across the dataset.")

    # 2. Skewed columns
    try:
        skewed = data.skew().sort_values(ascending=False)
        skewed_cols = skewed[abs(skewed) > 1].index.tolist()
        if skewed_cols:
            insights.append(f"{len(skewed_cols)} columns are highly skewed → {', '.join(skewed_cols[:5])}.")
    except Exception as e:
        insights.append(f"Could not analyze skewness: {str(e)}")

    # 3. Correlation insights
    try:
        corr = data.corr(numeric_only=True)
        if not corr.empty:
            high_corr = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                          .stack()
                          .sort_values(ascending=False))
            if not high_corr.empty:
                top_corr = high_corr.head(3)
                for (c1, c2), val in top_corr.items():
                    insights.append(f"High correlation ({val:.2f}) detected between {c1} and {c2}.")
    except Exception as e:
        insights.append(f"Could not analyze correlations: {str(e)}")

    # 4. Unique value counts (categorical columns)
    cat_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        unique_vals = data[col].nunique()
        if unique_vals < 5:
            insights.append(f"Column '{col}' has very few unique values ({unique_vals}), may be categorical encoding needed.")
    
    # Basic insights
    total_missing = data.isnull().sum().sum()
    missing_percentage = (total_missing / (data.shape[0] * data.shape[1])) * 100
    
    if missing_percentage > 10:
        insights.append(f"High missing data detected ({missing_percentage:.1f}%). Consider imputation strategies.")
    elif missing_percentage == 0:
        insights.append("✅ No missing values detected - excellent data quality!")
    
    # Numerical columns insights
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        # Check for potential outliers
        outlier_count = 0
        for col in numerical_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_count += outliers
        
        if outlier_count > 0:
            insights.append(f"Detected {outlier_count} potential outliers across numerical columns.")
    
    # Categorical columns insights
    categorical_cols = data.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        high_cardinality_cols = []
        for col in categorical_cols:
            if data[col].nunique() > data.shape[0] * 0.8:
                high_cardinality_cols.append(col)
        
        if high_cardinality_cols:
            insights.append(f"High cardinality categorical columns detected: {high_cardinality_cols}")
    
    # Correlation insights
    if len(numerical_cols) > 1:
        try:
            corr_matrix = data[numerical_cols].corr()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            if high_corr_pairs:
                insights.append(f"Found {len(high_corr_pairs)} highly correlated feature pairs - consider feature selection.")
        except Exception as e:
            insights.append(f"Could not analyze correlations: {str(e)}")
    
    # Data type insights
    memory_heavy_cols = []
    for col in data.columns:
        if data[col].dtype == 'object' and data[col].memory_usage(deep=True) > 1000000: # > 1MB
            memory_heavy_cols.append(col)
    
    if memory_heavy_cols:
        insights.append(f"Memory-heavy string columns detected: {memory_heavy_cols}. Consider categorical encoding.")
    
    # Sample size insights
    if data.shape[0] < 100:
        insights.append("⚠️ Small dataset detected. Results may not be statistically significant.")
    elif data.shape[0] > 100000:
        insights.append("📈 Large dataset detected. Consider sampling for exploratory analysis.")
    
    if not insights:
        insights.append("✅ Data appears to be in good condition for analysis!")
    
    return insights

def main():
    st.set_page_config(page_title="Advanced EDA", layout="wide")

    if 'df' not in st.session_state:
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'A': np.random.randn(100),
            'B': np.random.randn(100),
            'C': np.random.choice(['X', 'Y', 'Z'], 100)
        })
        st.session_state['df'] = sample_data
    
    show_advanced_eda()

if __name__ == "__main__":
    main()