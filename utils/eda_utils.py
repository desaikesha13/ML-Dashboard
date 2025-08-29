import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import median_abs_deviation
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')

def generate_data_summary(df):
    summary = df.describe(include='all').transpose()
    summary['dtype'] = df.dtypes
    summary['missing_count'] = df.isnull().sum()
    summary['missing_percentage'] = (df.isnull().sum() / len(df)) * 100
    return summary

def create_correlation_heatmap(df_numeric):
    corr = df_numeric.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr.values,
        texttemplate='%{text:.2f}',
        showscale=True
    ))
    fig.update_layout(
        title='Correlation Heatmap',
        xaxis_title='Features',
        yaxis_title='Features',
        height=600
    )
    return fig

def create_distribution_plots(df, columns):
    plots = []
    for col in columns:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[col], name=col, nbinsx=30))
        fig.update_layout(
            title=f'Distribution of {col}',
            xaxis_title=col,
            yaxis_title='Frequency',
            height=400
        )
        plots.append(fig)
    return plots

def detect_outliers(df, column, method="IQR"):
    """Detect outliers and return a plotly figure"""
    if method == "IQR":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
    elif method == "Z-Score":
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        outliers = df[z_scores > 3].index
    elif method == "Modified Z-Score":
        median = df[column].median()
        mad = median_abs_deviation(df[column].dropna())
        modified_z_scores = 0.6745 * (df[column] - median) / mad
        outliers = df[np.abs(modified_z_scores) > 3.5].index
    
    # Create a box plot to visualize outliers
    fig = go.Figure()
    fig.add_trace(go.Box(y=df[column], name=column, boxpoints='outliers'))
    fig.update_layout(
        title=f'Outlier Detection for {column} ({method} method)',
        xaxis_title='Column',
        yaxis_title='Value',
        height=400
    )
    return fig

def missing_values_analysis(df):
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    if len(missing_data) == 0:
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(x=missing_data.index, y=missing_data.values, name='Missing Values'))
    fig.update_layout(
        title='Missing Values by Column',
        xaxis_title='Columns',
        yaxis_title='Missing Count',
        height=400
    )
    return fig

def categorical_analysis(df, column):
    value_counts = df[column].value_counts()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=value_counts.index, y=value_counts.values, name=column))
    fig.update_layout(
        title=f'Distribution of {column}',
        xaxis_title=column,
        yaxis_title='Count',
        height=400
    )
    return fig

def data_quality_report(df: pd.DataFrame):
    """Generate a simple data quality report."""
    st.subheader("📋 Data Quality Report")
    
    report = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.values,
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })

    st.dataframe(report, use_container_width=True)

class AdvancedEDA:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def outlier_detection(self):
        st.write("### Advanced Outlier Detection")
        numeric_cols = self.data.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if numeric_cols:
            col = st.selectbox("Select column", numeric_cols, key="adv_outliers")
            method = st.radio("Method", ["IQR", "Z-Score", "Modified Z-Score"])
            outliers = detect_outliers(self.data, col, method)
            st.write(f"Found {len(outliers)} outliers in `{col}` using {method} method.")

    def feature_importance_analysis(self):
        st.write("### Feature Importance (Random Forest)")
        numeric_cols = self.data.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = self.data.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            le = LabelEncoder()
            for col in cat_cols:
                self.data[col] = le.fit_transform(self.data[col].astype(str))
        target = st.selectbox("Select Target Variable", self.data.columns)
        features = [col for col in self.data.columns if col != target]
        if len(features) > 0:
            try:
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(self.data[features], self.data[target])
                importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
                st.bar_chart(importances)
            except Exception as e:
                st.error(f"Error in feature importance: {e}")
        else:
            st.warning("Not enough features for analysis.")

    def dimensionality_reduction(self):
        st.write("### Dimensionality Reduction (PCA)")
        numeric_cols = self.data.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numerical columns for PCA.")
            return
        pca = PCA(n_components=2)
        components = pca.fit_transform(self.data[numeric_cols].dropna())
        pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
        st.write("PCA Result (first 5 rows):")
        st.dataframe(pca_df.head())
        fig, ax = plt.subplots()
        sns.scatterplot(x="PC1", y="PC2", data=pca_df, ax=ax)
        st.pyplot(fig)

    def statistical_tests(self):
        st.write("### Statistical Tests")
        numeric_cols = self.data.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numerical columns for t-test.")
            return
        col1 = st.selectbox("Select First Column", numeric_cols, key="ttest1")
        col2 = st.selectbox("Select Second Column", numeric_cols, key="ttest2")
        if col1 != col2:
            t_stat, p_val = ttest_ind(self.data[col1].dropna(), self.data[col2].dropna())
            st.write(f"T-test between `{col1}` and `{col2}` → t={t_stat:.3f}, p={p_val:.3f}")
        else:
            st.info("Please select two different columns.")