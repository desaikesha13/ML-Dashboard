import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import pages.advanced_eda as advanced_Eda
import pages.model_comparison as model_comparison
from utils import eda_utils, ml_utils 

# Page configuration
st.set_page_config(
    page_title="AutoEDA ML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/autoeda-ml-dashboard',
        'Report a bug': 'https://github.com/your-repo/autoeda-ml-dashboard/issues',
        'About': "# AutoEDA ML Dashboard\nAn interactive tool for automated exploratory data analysis and machine learning model comparison."
    }
)

# Import custom modules
try:
    from utils.eda_utils import (
        generate_data_summary, create_correlation_heatmap, 
        create_distribution_plots, detect_outliers, 
        missing_values_analysis, categorical_analysis
    )
    from utils.ml_utils import (
        quick_model_preview, feature_importance_analysis,
        data_preprocessing_suggestions
    )
    from pages.advanced_eda import show_advanced_eda
    from pages.model_comparison import show_model_comparison
except ImportError as e:
    st.error(f"⚠️ Import Error: {e}")
    st.info("Please ensure all utility modules are properly set up in the utils/ and pages/ directories.")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    
    .uploadedFile {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f0f2f6, #ffffff);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'data_uploaded' not in st.session_state:
        st.session_state.data_uploaded = False
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "home"
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

def load_sample_data():
    """Load sample datasets for demonstration"""
    sample_datasets = {
        "Iris Dataset": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        "Tips Dataset": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
        "Titanic Dataset": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
        "Boston Housing": "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    }
    
    st.sidebar.header("📊 Sample Datasets")
    selected_sample = st.sidebar.selectbox(
        "Try with sample data:",
        ["None"] + list(sample_datasets.keys())
    )
    
    if selected_sample != "None" and st.sidebar.button("Load Sample Dataset"):
        try:
            with st.spinner(f"Loading {selected_sample}..."):
                df = pd.read_csv(sample_datasets[selected_sample])
                st.session_state.df = df
                st.session_state.data_uploaded = True
                st.success(f"✅ {selected_sample} loaded successfully!")
                st.rerun()
        except Exception as e:
            st.error(f"❌ Error loading sample dataset: {str(e)}")

def upload_data():
    """Handle data upload functionality"""
    st.header("📁 Data Upload")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'json'],
        help="Supported formats: CSV, Excel (XLSX), JSON"
    )
    
    if uploaded_file is not None:
        try:
            # Determine file type and read accordingly
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            
            # Store in session state
            st.session_state["df"] = df
            st.session_state.data_uploaded = True
            
            st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Size", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # Preview data
            st.subheader("🔍 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Please ensure your file is properly formatted and not corrupted.")

def show_data_overview():
    """Display comprehensive data overview"""
    if st.session_state.df is None:
        st.warning("⚠️ No data uploaded yet!")
        return
    
    df = st.session_state.df
    
    st.header("📋 Data Overview")

    # Basic statistics
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Dataset Summary")
        summary_stats = eda_utils.generate_data_summary(df)
        st.dataframe(summary_stats, use_container_width=True)

    with col2:
        st.subheader("🔢 Quick Stats")

        # CSS styling for metric cards
        st.markdown("""
            <style>
            .metric-card {
                background-color: #262730; /* dark card background */
                padding: 12px 15px;
                border-radius: 12px;
                margin-bottom: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                color: white; /* text visible in dark mode */
            }
            .metric-card h4 {
                margin: 0 0 6px 0;
                color: #FFD700; /* gold headings */
            }
            .metric-card p {
                margin: 3px 0;
                font-size: 14px;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <h4>Dataset Dimensions</h4>
            <p><strong>Rows:</strong> {df.shape[0]:,}</p>
            <p><strong>Columns:</strong> {df.shape[1]:,}</p>
        </div>

        <div class="metric-card">
            <h4>Data Types</h4>
            <p><strong>Numeric:</strong> {len(df.select_dtypes(include=[np.number]).columns)}</p>
            <p><strong>Categorical:</strong> {len(df.select_dtypes(include=['object']).columns)}</p>
            <p><strong>DateTime:</strong> {len(df.select_dtypes(include=['datetime64']).columns)}</p>
        </div>

        <div class="metric-card">
            <h4>Data Quality</h4>
            <p><strong>Missing Values:</strong> {df.isnull().sum().sum():,}</p>
            <p><strong>Duplicates:</strong> {df.duplicated().sum():,}</p>
            <p><strong>Memory Usage:</strong> {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB</p>
        </div>
        """, unsafe_allow_html=True)

    # Data types breakdown
    st.subheader("🔍 Column Information")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2),
        'Unique Values': df.nunique(),
        'Sample Value': [df[col].dropna().iloc[0] if not df[col].dropna().empty else 'NaN' for col in df.columns]
    })
    st.dataframe(col_info, use_container_width=True)

def show_quick_eda():
    if st.session_state.df is None:
        st.warning("⚠️ No data uploaded yet!")
        return
    
    df = st.session_state.df
    
    st.header("🚀 Quick EDA")
    
    # Missing values analysis
    st.subheader("❓ Missing Values Analysis")
    missing_analysis = eda_utils.missing_values_analysis(df)
    # 💡 FIX: Add a more robust check for a valid Plotly Figure
    if isinstance(missing_analysis, go.Figure):
        st.plotly_chart(missing_analysis, use_container_width=True)
    elif isinstance(missing_analysis, list) and all(isinstance(fig, go.Figure) for fig in missing_analysis):
        for fig in missing_analysis:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No missing values or no plot could be generated.")
    
    # Correlation heatmap for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        st.subheader("🔥 Correlation Heatmap")
        corr_heatmap = eda_utils.create_correlation_heatmap(df[numeric_cols])
        if isinstance(corr_heatmap, go.Figure):
            st.plotly_chart(corr_heatmap, use_container_width=True)
        else:
            st.info("Correlation heatmap could not be generated.")

    # Distribution plots
    if len(numeric_cols) > 0:
        st.subheader("📈 Distribution Analysis")
        selected_columns = st.multiselect(
            "Select columns for distribution analysis:",
            numeric_cols.tolist(),
            default=numeric_cols.tolist()[:4]
        )
        
        if selected_columns:
            dist_plots = eda_utils.create_distribution_plots(df, selected_columns)
            if isinstance(dist_plots, list) and all(isinstance(fig, go.Figure) for fig in dist_plots):
                for fig in dist_plots:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Distribution plots could not be generated for the selected columns.")
    
    # Categorical analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        st.subheader("📊 Categorical Analysis")
        selected_cat = st.selectbox(
            "Select categorical column:",
            categorical_cols.tolist()
        )
        
        if selected_cat:
            cat_analysis = eda_utils.categorical_analysis(df, selected_cat)
            if isinstance(cat_analysis, go.Figure):
                st.plotly_chart(cat_analysis, use_container_width=True)
            else:
                st.info(f"Categorical analysis plot could not be generated for '{selected_cat}'.")
    
    # Outlier detection
    if len(numeric_cols) > 0:
        st.subheader("🎯 Outlier Detection")
        outlier_col = st.selectbox(
            "Select column for outlier detection:",
            numeric_cols.tolist()
        )
        
        if outlier_col:
            outlier_fig = eda_utils.detect_outliers(df, outlier_col)
            if isinstance(outlier_fig, go.Figure):
                st.plotly_chart(outlier_fig, use_container_width=True)
            else:
                st.info(f"Outlier detection plot could not be generated for '{outlier_col}'.")

def show_ml_preview():
    """Display ML model preview and suggestions"""
    if st.session_state.df is None:
        st.warning("⚠️ No data uploaded yet!")
        return
    
    df = st.session_state.df
    
    st.header("🤖 ML Preview & Suggestions")
    
    # Data preprocessing suggestions
    st.subheader("🔧 Data Preprocessing Suggestions")
    preprocessing_suggestions = ml_utils.data_preprocessing_suggestions(df)
    if preprocessing_suggestions:
        for suggestion in preprocessing_suggestions:
            st.info(f"💡 {suggestion}")
    else:
        st.info("No preprocessing suggestions available.")
    
    # Quick model preview
    st.subheader("⚡ Quick Model Preview")
    
    # Select target variable
    target_column = st.selectbox(
        "Select target variable for quick analysis:",
        df.columns.tolist()
    )
    
    if target_column:
        # Determine problem type
        if df[target_column].dtype == 'object' or df[target_column].nunique() < 10:
            problem_type = "Classification"
        else:
            problem_type = "Regression"
        
        st.info(f"📊 Detected problem type: **{problem_type}**")
        
        # Feature selection
        feature_columns = [col for col in df.columns if col != target_column]
        selected_features = st.multiselect(
            "Select features:",
            feature_columns,
            default=feature_columns[:5]  # Default to first 5 features
        )
        
        if selected_features:
            if st.button("🚀 Run Quick Model Preview"):
                with st.spinner("Training models..."):
                    preview_results = ml_utils.quick_model_preview(df, target_column, selected_features, problem_type)
                    if preview_results:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("📊 Model Performance")
                            st.json(preview_results['performance'])
                        with col2:
                            st.subheader("🎯 Feature Importance")
                            importance_fig = ml_utils.feature_importance_analysis(
                                preview_results['model'], 
                                selected_features
                            )
                            st.plotly_chart(importance_fig, use_container_width=True)

def main_sidebar():
    """Create main sidebar navigation"""
    st.sidebar.title("🤖 AutoEDA ML Dashboard")
    
    # Navigation
    pages = {
        "🏠 Home": "home",
        "📊 Data Overview": "overview", 
        "🚀 Quick EDA": "eda",
        "🔬 Advanced EDA": "advanced_eda",
        "🤖 ML Preview": "ml_preview",
        "📈 Model Comparison": "model_comparison"
    }
    
    selected_page = st.sidebar.radio("Navigate to:", list(pages.keys()))
    st.session_state.current_page = pages[selected_page]
    
    # Data info sidebar
    if st.session_state.data_uploaded and st.session_state.df is not None:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📋 Current Dataset")
        df = st.session_state.df
        st.sidebar.write(f"**Rows:** {df.shape[0]:,}")
        st.sidebar.write(f"**Columns:** {df.shape[1]:,}")
        st.sidebar.write(f"**Size:** {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Clear data button
        if st.sidebar.button("🗑️ Clear Data"):
            st.session_state.df = None
            st.session_state.data_uploaded = False
            st.session_state.analysis_complete = False
            st.rerun()
    
    # Sample data loader
    load_sample_data()
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info(
        "AutoEDA ML Dashboard combines automated exploratory data analysis "
        "with interactive machine learning model comparison. Upload your data "
        "and get instant insights!"
    )

def show_home():
    """Show home page"""
    st.markdown('<h1 class="main-header">AutoEDA ML Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the **AutoEDA ML Dashboard** - your one-stop solution for automated exploratory data analysis 
    and machine learning model comparison! 🎉
    
    ## 🌟 Features
    
    - **📊 Automated EDA**: Get instant insights from your data
    - **🔍 Advanced Analysis**: Deep dive into data patterns and relationships  
    - **🤖 ML Model Comparison**: Compare multiple models side-by-side
    - **📈 Interactive Visualizations**: Explore data with interactive plots
    - **💡 Smart Recommendations**: Get AI-powered insights and suggestions
    
    ## 🚀 Getting Started
    
    1. **Upload your data** using the file uploader below
    2. **Explore** your data with automated EDA
    3. **Compare** machine learning models
    4. **Export** your findings and models
    
    ---
    """)
    
    # Upload section
    upload_data()
    
    # Quick stats if data is loaded
    if st.session_state.data_uploaded:
        st.markdown("---")
        st.subheader("✨ Ready to Explore!")
        st.success("Your data is loaded and ready for analysis. Use the sidebar to navigate to different sections.")
        
        # Quick action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📊 View Data Overview", use_container_width=True):
                st.session_state.current_page = "overview"
                st.rerun()
        with col2:
            if st.button("🚀 Quick EDA", use_container_width=True):
                st.session_state.current_page = "eda"
                st.rerun()
        with col3:
            if st.button("🤖 Compare Models", use_container_width=True):
                st.session_state.current_page = "model_comparison"
                st.rerun()

def main():
    """Main application function"""
    initialize_session_state()
    main_sidebar()
    
    # Route to appropriate page
    if st.session_state.current_page == "home":
        show_home()
    elif st.session_state.current_page == "overview":
        show_data_overview()
    elif st.session_state.current_page == "eda":
        show_quick_eda()
    elif st.session_state.current_page == "advanced_eda":
        try:
            show_advanced_eda()
        except NameError:
            st.error("Advanced EDA module not found. Please ensure advanced_eda.py is in the pages/ directory.")
    elif st.session_state.current_page == "ml_preview":
        show_ml_preview()
    elif st.session_state.current_page == "model_comparison":
        try:
            show_model_comparison()
        except NameError:
            st.error("Model comparison module not found. Please ensure model_comparison.py is in the pages/ directory.")
    else:
        show_home()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Made with ❤️ using Streamlit | "
        "AutoEDA ML Dashboard v1.0 | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

if __name__ == "__main__":
    main()