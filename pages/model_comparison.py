import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

def show_model_comparison():
    """Main function to display the model comparison page"""
    st.title("🤖 Model Comparison Dashboard")
    st.markdown("---")

    # --- Check if dataset exists in session state (support both df and data) ---
    df = None
    if "df" in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
    elif "data" in st.session_state and st.session_state.data is not None:
        df = st.session_state.data

    if df is None:
        st.warning("⚠️ Please upload data first from the **main page**!")
        st.info("👉 Go to the main page and upload your dataset to start model comparison.")
        return

    # --- Show dataset preview so page is never blank ---
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # --- Sidebar configuration ---
    st.sidebar.header("🔧 Model Configuration")

    # Problem type selection
    problem_type = st.sidebar.selectbox(
        "Select Problem Type:",
        ["Classification", "Regression"]
    )

    # Target variable selection
    target_column = st.sidebar.selectbox(
        "Select Target Variable:",
        df.columns.tolist()
    )
    
    # Detect target type
    try:
        from sklearn.utils.multiclass import type_of_target
        target_type = type_of_target(df[target_column])

        if problem_type == "Classification" and target_type == "continuous":
            st.error("❌ Target column seems continuous. Please switch to Regression.")
            return
        elif problem_type == "Regression" and target_type in ["binary", "multiclass"]:
            st.error("❌ Target column seems categorical. Please switch to Classification.")
            return
    except Exception as e:
        st.warning(f"⚠️ Could not detect target type: {str(e)}. Proceeding with user selection.")

    # Feature selection
    feature_columns = st.sidebar.multiselect(
        "Select Features:",
        [col for col in df.columns if col != target_column],
        default=[col for col in df.columns if col != target_column][:10]  # Default first 10
    )

    if not feature_columns:
        st.warning("⚠️ Please select at least one feature!")
        return

    # Test size selection
    test_size = st.sidebar.slider(
        "Test Size (%):", min_value=10, max_value=50, value=20
    ) / 100

    # Cross-validation folds
    cv_folds = st.sidebar.slider(
        "Cross-Validation Folds:", min_value=3, max_value=10, value=5
    )

    # Model selection
    if problem_type == "Classification":
        available_models = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "SVM": SVC(random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Naive Bayes": GaussianNB(),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }
    else:
        available_models = {
            "Random Forest": RandomForestRegressor(random_state=42),
            "Linear Regression": LinearRegression(),
            "SVM": SVR(),
            "K-Nearest Neighbors": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        }

    selected_models = st.sidebar.multiselect(
        "Select Models to Compare:",
        list(available_models.keys()),
        default=list(available_models.keys())[:3]
    )

    if not selected_models:
        st.warning("⚠️ Please select at least one model!")
        return

    # --- Always show instructions before running ---
    st.info("👈 Configure options in the sidebar and then click **🚀 Start Model Comparison**.")

    # Start comparison button
    if st.sidebar.button("🚀 Start Model Comparison", type="primary"):
        with st.spinner("Training and comparing models..."):
            try:
                results = run_model_comparison(
                    df, target_column, feature_columns, problem_type,
                    selected_models, available_models, test_size, cv_folds
                )
                if results:
                    display_comparison_results(results, problem_type)
                else:
                    st.error("⚠️ No results returned. Please check your configuration.")
            except Exception as e:
                st.error(f"⚠️ Error during model comparison: {e}")

def prepare_data(df, target_column, feature_columns):
    try:
        # Create a copy of the dataframe
        data = df.copy()
        
        # Select only the columns we need
        columns_needed = feature_columns + [target_column]
        data = data[columns_needed]
        
        # Drop rows with missing values in target column first
        data = data.dropna(subset=[target_column])
        
        # Handle missing values in features
        for col in feature_columns:
            if data[col].dtype == 'object':
                mode_value = data[col].mode()
                fill_value = mode_value[0] if not mode_value.empty else 'Unknown'
                data[col].fillna(fill_value, inplace=True)
            else:
                # Handle case where median might be NaN
                median_val = data[col].median()
                if pd.isna(median_val):
                    data[col].fillna(0, inplace=True)  # Use 0 as fallback
                else:
                    data[col].fillna(median_val, inplace=True)
        
        # Final check - drop any remaining NaN rows
        data = data.dropna()
        
        # Ensure we have enough data
        try:
            data_length = len(data) if hasattr(data, '__len__') else 0
            if data_length < 10:
                raise ValueError(f"Not enough data after preprocessing (minimum 10 samples required, got {data_length})")
        except Exception as e:
            raise ValueError(f"Error checking data length: {str(e)}")
        
        # Prepare features and target
        X = data[feature_columns]
        y = data[target_column]
        
        # Handle categorical features
        try:
            categorical_cols = X.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        except Exception as e:
            print(f"Warning: Error encoding categorical features: {str(e)}")
        
        # Handle categorical target for classification
        try:
            if y.dtype == 'object':
                le_target = LabelEncoder()
                y = le_target.fit_transform(y)
        except Exception as e:
            print(f"Warning: Error encoding target variable: {str(e)}")
        
        print(f"Data shape after preprocessing: X={X.shape}, y={y.shape}")
        return X, y
        
    except Exception as e:
        print(f"Error in prepare_data: {str(e)}")
        raise e

def run_model_comparison(df, target_column, feature_columns, problem_type, 
                        selected_models, available_models, test_size, cv_folds):
    try:
        # Validate inputs
        if not feature_columns or len(feature_columns) == 0:
            st.error("❌ No features selected!")
            return None
            
        if target_column in feature_columns:
            st.error("❌ Target column cannot be in feature columns!")
            return None
        
        # Prepare data
        X, y = prepare_data(df, target_column, feature_columns)
        
        # Ensure we have data
        if X.empty or len(y) == 0:
            st.error("❌ No data remaining after preprocessing!")
            return None
            
        # Ensure we have enough data for training
        try:
            X_length = len(X) if hasattr(X, '__len__') else 0
            if X_length < 10:
                st.error(f"❌ Not enough data for training (minimum 10 samples required, got {X_length})")
                return None
        except Exception as e:
            st.error(f"❌ Error checking data length: {str(e)}")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if problem_type == "Classification" else None
        )
        
        print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
        print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
        
        # Scale features for models that need it
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        
        for model_name in selected_models:
            try:
                model = available_models[model_name]
                
                # Use scaled data for models that benefit from scaling
                if model_name in ["SVM", "Logistic Regression", "K-Nearest Neighbors"]:
                    X_train_use = X_train_scaled
                    X_test_use = X_test_scaled
                else:
                    X_train_use = X_train
                    X_test_use = X_test
                
                # Train model
                model.fit(X_train_use, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_use)
                
                # CRITICAL FIX: Ensure both arrays have same length
                if len(y_test) != len(y_pred):
                    st.warning(f"⚠️ Shape mismatch for {model_name}: y_test={len(y_test)}, y_pred={len(y_pred)}")
                    min_len = min(len(y_test), len(y_pred))
                    y_test_trimmed = y_test.iloc[:min_len] if hasattr(y_test, 'iloc') else y_test[:min_len]
                    y_pred_trimmed = y_pred[:min_len]
                else:
                    y_test_trimmed = y_test
                    y_pred_trimmed = y_pred
                
                # Convert to consistent format
                y_test_array = np.array(y_test_trimmed)
                y_pred_array = np.array(y_pred_trimmed)
                
                print(f"{model_name} - Final shapes: y_test={y_test_array.shape}, y_pred={y_pred_array.shape}")
                
                # Cross-validation
                try:
                    if model_name in ["SVM", "Logistic Regression", "K-Nearest Neighbors"]:
                        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds)
                    else:
                        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds)
                except Exception as e:
                    print(f"Warning: Cross-validation failed for {model_name}: {str(e)}")
                    cv_scores = np.array([0.0])  # Default fallback
                
                # Calculate metrics
                try:
                    if problem_type == "Classification":
                        metrics = {
                            'accuracy': accuracy_score(y_test_array, y_pred_array),
                            'precision': precision_score(y_test_array, y_pred_array, average='weighted', zero_division=0),
                            'recall': recall_score(y_test_array, y_pred_array, average='weighted', zero_division=0),
                            'f1': f1_score(y_test_array, y_pred_array, average='weighted', zero_division=0),
                            'cv_mean': cv_scores.mean(),
                            'cv_std': cv_scores.std()
                        }
                        
                        # ROC curve data for binary classification
                        try:
                            # Ensure y is a numpy array for unique calculation
                            y_array = np.array(y) if not isinstance(y, np.ndarray) else y
                            if len(np.unique(y_array)) == 2:
                                if hasattr(model, "predict_proba"):
                                    y_prob = model.predict_proba(X_test_use)[:, 1]
                                    
                                    # Ensure probability array matches test set size
                                    if len(y_prob) != len(y_test_array):
                                        min_len = min(len(y_prob), len(y_test_array))
                                        y_prob = y_prob[:min_len]
                                        y_test_for_roc = y_test_array[:min_len]
                                    else:
                                        y_test_for_roc = y_test_array
                                    
                                    fpr, tpr, _ = roc_curve(y_test_for_roc, y_prob)
                                    metrics['roc_auc'] = auc(fpr, tpr)
                                    metrics['fpr'] = fpr.tolist()
                                    metrics['tpr'] = tpr.tolist()
                        except Exception as e:
                            print(f"Warning: Could not calculate ROC curve for {model_name}: {str(e)}")
                            # Continue without ROC curve data
                    else:  # Regression
                        metrics = {
                            'mse': mean_squared_error(y_test_array, y_pred_array),
                            'mae': mean_absolute_error(y_test_array, y_pred_array),
                            'r2': r2_score(y_test_array, y_pred_array),
                            'rmse': np.sqrt(mean_squared_error(y_test_array, y_pred_array)),
                            'cv_mean': cv_scores.mean(),
                            'cv_std': cv_scores.std()
                        }
                except Exception as e:
                    print(f"Warning: Metrics calculation failed for {model_name}: {str(e)}")
                    # Provide default metrics
                    metrics = {
                        'accuracy': 0.0 if problem_type == "Classification" else None,
                        'precision': 0.0 if problem_type == "Classification" else None,
                        'recall': 0.0 if problem_type == "Classification" else None,
                        'f1': 0.0 if problem_type == "Classification" else None,
                        'mse': 0.0 if problem_type == "Regression" else None,
                        'mae': 0.0 if problem_type == "Regression" else None,
                        'r2': 0.0 if problem_type == "Regression" else None,
                        'rmse': 0.0 if problem_type == "Regression" else None,
                        'cv_mean': 0.0,
                        'cv_std': 0.0
                    }
                
                # Save results with consistent array conversion
                try:
                    # Ensure metrics contains all required keys
                    if problem_type == "Classification":
                        required_keys = ['accuracy', 'precision', 'recall', 'f1', 'cv_mean', 'cv_std']
                    else:
                        required_keys = ['mse', 'mae', 'r2', 'rmse', 'cv_mean', 'cv_std']
                    
                    # Check if all required metrics are present
                    missing_keys = [key for key in required_keys if key not in metrics]
                    if missing_keys:
                        print(f"Warning: Missing metrics for {model_name}: {missing_keys}")
                        # Fill missing metrics with default values
                        for key in missing_keys:
                            if key in ['accuracy', 'precision', 'recall', 'f1', 'r2']:
                                metrics[key] = 0.0
                            elif key in ['mse', 'mae', 'rmse']:
                                metrics[key] = float('inf')
                            elif key in ['cv_mean', 'cv_std']:
                                metrics[key] = 0.0
                    
                    results[model_name] = {
                        'model': model,
                        'metrics': metrics,
                        'predictions': pd.Series(y_pred_array),
                        'actual': pd.Series(y_test_array)
                    }
                except Exception as e:
                    print(f"Error saving results for {model_name}: {str(e)}")
                    # Save with default metrics
                    default_metrics = {
                        'accuracy': 0.0 if problem_type == "Classification" else None,
                        'precision': 0.0 if problem_type == "Classification" else None,
                        'recall': 0.0 if problem_type == "Classification" else None,
                        'f1': 0.0 if problem_type == "Classification" else None,
                        'mse': 0.0 if problem_type == "Regression" else None,
                        'mae': 0.0 if problem_type == "Regression" else None,
                        'r2': 0.0 if problem_type == "Regression" else None,
                        'rmse': 0.0 if problem_type == "Regression" else None,
                        'cv_mean': 0.0,
                        'cv_std': 0.0
                    }
                    results[model_name] = {
                        'model': model,
                        'metrics': default_metrics,
                        'predictions': pd.Series(y_pred_array),
                        'actual': pd.Series(y_test_array)
                    }
                
            except Exception as e:
                st.error(f"❌ Error training {model_name}: {str(e)}")
                continue
        
        return results
    
    except Exception as e:
        st.error(f"Error in model comparison: {str(e)}")
        import traceback
        st.code(traceback.format_exc()) 
        return None

def display_comparison_results(results, problem_type):
    """Display model comparison results with comprehensive error handling"""
    st.success("✅ Model comparison completed!")
    
    try:
        # Validate results
        if not results or not isinstance(results, dict):
            st.error("❌ Invalid results data.")
            return
        
        # Create metrics comparison dataframe with robust error handling
        metrics_df = pd.DataFrame()
        for model_name, result in results.items():
            try:
                if 'metrics' in result and result['metrics'] and isinstance(result['metrics'], dict):
                    metrics_row = result['metrics'].copy()
                    metrics_row['Model'] = model_name
                    metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics_row])], ignore_index=True)
            except Exception as e:
                print(f"Warning: Could not process metrics for {model_name}: {str(e)}")
                continue
        
        if metrics_df.empty:
            st.error("❌ No metrics data available for display.")
            return
        
        # Set Model as index
        try:
            metrics_df.set_index('Model', inplace=True)
        except Exception as e:
            st.error(f"❌ Error setting index: {str(e)}")
            return
        
        # Rename columns for readability - only rename columns that exist
        rename_map = {
            "accuracy": "Accuracy",
            "precision": "Precision", 
            "recall": "Recall",
            "f1": "F1-Score",
            "cv_mean": "CV Mean",
            "cv_std": "CV Std",
            "r2": "R²",
            "rmse": "RMSE",
            "mae": "MAE",
            "roc_auc": "ROC-AUC"
        }
        
        # Only rename columns that actually exist in the dataframe
        existing_columns = metrics_df.columns.tolist()
        columns_to_rename = {k: v for k, v in rename_map.items() if k in existing_columns}
        if columns_to_rename:
            try:
                metrics_df.rename(columns=columns_to_rename, inplace=True)
            except Exception as e:
                print(f"Warning: Could not rename columns: {str(e)}")
        
        print(f"Available columns after renaming: {metrics_df.columns.tolist()}")
        
    except Exception as e:
        st.error(f"❌ Error creating metrics dataframe: {str(e)}")
        return
    
    # Display metrics table with highlights
    st.subheader("📊 Model Performance Comparison")
    try:
        # Try to apply highlighting with error handling
        numeric_df = metrics_df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            styled_df = numeric_df.round(4).style.highlight_max(axis=0, color="lightgreen").highlight_min(axis=0, color="lightcoral")
            st.dataframe(styled_df)
        else:
            st.dataframe(metrics_df.round(4))
    except Exception as e:
        # Fallback to simple dataframe if highlighting fails
        print(f"Warning: Could not apply dataframe highlighting: {str(e)}")
        st.dataframe(metrics_df.round(4))
    
    # Parallel Coordinates Plot for holistic view
    try:
        st.subheader("🌐 Parallel Coordinates View")
        from pandas.plotting import parallel_coordinates
        import matplotlib.pyplot as plt
        
        # Only keep numeric columns
        numeric_cols = metrics_df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 1:
            metrics_for_plot = metrics_df[numeric_cols].reset_index()
            fig, ax = plt.subplots(figsize=(10, 6))
            parallel_coordinates(metrics_for_plot, "Model", colormap="viridis")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.info("Not enough numeric metrics to plot parallel coordinates.")
    except Exception as e:
        st.warning(f"⚠️ Could not display parallel coordinates: {str(e)}")
    
    # Side-by-side visualizations
    col1, col2 = st.columns(2)
    
    if problem_type == "Classification":
        # Performance metrics bar chart
        with col1:
            try:
                st.subheader("📈 Performance Metrics")
                fig = go.Figure()
                
                # Try both uppercase and lowercase metric names
                metrics_to_plot = [
                    ('Accuracy', 'accuracy'),
                    ('Precision', 'precision'),
                    ('Recall', 'recall'),
                    ('F1-Score', 'f1')
                ]
                
                for display_name, col_name in metrics_to_plot:
                    if display_name in metrics_df.columns:
                        metric_col = display_name
                    elif col_name in metrics_df.columns:
                        metric_col = col_name
                    else:
                        continue
                    
                    # Ensure the column contains numeric data
                    if pd.api.types.is_numeric_dtype(metrics_df[metric_col]):
                        fig.add_trace(go.Bar(
                            name=display_name,
                            x=metrics_df.index,
                            y=metrics_df[metric_col],
                            text=metrics_df[metric_col].round(3),
                            textposition='auto',
                        ))
                
                fig.update_layout(
                    title="Classification Metrics Comparison",
                    xaxis_title="Models",
                    yaxis_title="Score",
                    barmode='group',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Could not display performance metrics: {str(e)}")
        
        # Cross-validation scores
        with col2:
            try:
                st.subheader("🎯 Cross-Validation Scores")
                fig = go.Figure()
                
                # Try both uppercase and lowercase CV column names
                cv_mean_col = None
                cv_std_col = None
                
                if "CV Mean" in metrics_df.columns:
                    cv_mean_col = "CV Mean"
                    cv_std_col = "CV Std" if "CV Std" in metrics_df.columns else None
                elif "cv_mean" in metrics_df.columns:
                    cv_mean_col = "cv_mean"
                    cv_std_col = "cv_std" if "cv_std" in metrics_df.columns else None
                
                if cv_mean_col and pd.api.types.is_numeric_dtype(metrics_df[cv_mean_col]):
                    error_data = None
                    if cv_std_col and pd.api.types.is_numeric_dtype(metrics_df[cv_std_col]):
                        error_data = dict(type='data', array=metrics_df[cv_std_col])
                    
                    fig.add_trace(go.Bar(
                        name='CV Mean',
                        x=metrics_df.index,
                        y=metrics_df[cv_mean_col],
                        error_y=error_data,
                        text=metrics_df[cv_mean_col].round(3),
                        textposition='auto',
                    ))
                
                fig.update_layout(
                    title="Cross-Validation Performance",
                    xaxis_title="Models",
                    yaxis_title="CV Score",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Could not display cross-validation scores: {str(e)}")
        
        # ROC Curves
        try:
            # Check if any models have ROC curve data
            models_with_roc = []
            for model_name, result in results.items():
                if ('fpr' in result['metrics'] and 'tpr' in result['metrics'] and 
                    'roc_auc' in result['metrics'] and 
                    isinstance(result['metrics']['fpr'], (list, np.ndarray)) and
                    isinstance(result['metrics']['tpr'], (list, np.ndarray))):
                    models_with_roc.append(model_name)
            
            if models_with_roc:
                st.subheader("📉 ROC Curves")
                fig = go.Figure()
                
                for model_name in models_with_roc:
                    try:
                        result = results[model_name]
                        fpr = np.array(result['metrics']['fpr'])
                        tpr = np.array(result['metrics']['tpr'])

                        # Ensure equal lengths
                        min_len = min(len(fpr), len(tpr))
                        fpr_corrected = fpr[:min_len]
                        tpr_corrected = tpr[:min_len]
                        
                        # Get ROC-AUC value safely
                        roc_auc_value = result['metrics'].get('roc_auc', 0.0)
                        if isinstance(roc_auc_value, (int, float)):
                            auc_display = f"{roc_auc_value:.3f}"
                        else:
                            auc_display = "N/A"
                        
                        fig.add_trace(go.Scatter(
                            x=fpr_corrected,  
                            y=tpr_corrected,
                            mode='lines',
                            name=f"{model_name} (AUC: {auc_display})"
                        ))
                    except Exception as e:
                        print(f"Warning: Could not plot ROC curve for {model_name}: {str(e)}")
                        continue
                
                # Add diagonal baseline
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    line=dict(dash='dash', color='gray'),
                    name='Random Classifier'
                ))
                
                fig.update_layout(
                    title="ROC Curves Comparison",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ℹ️ No ROC curves available. This typically happens for multi-class classification or when models don't support probability predictions.")
        except Exception as e:
            st.warning(f"⚠️ Could not display ROC curves: {str(e)}")
    
    else:  # Regression
        with col1:
            try:
                st.subheader("📈 Regression Metrics")
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('R² Score', 'RMSE', 'MAE', 'Cross-Validation'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Try both uppercase and lowercase metric names
                r2_col = "R²" if "R²" in metrics_df.columns else ("r2" if "r2" in metrics_df.columns else None)
                rmse_col = "RMSE" if "RMSE" in metrics_df.columns else ("rmse" if "rmse" in metrics_df.columns else None)
                mae_col = "MAE" if "MAE" in metrics_df.columns else ("mae" if "mae" in metrics_df.columns else None)
                cv_mean_col = "CV Mean" if "CV Mean" in metrics_df.columns else ("cv_mean" if "cv_mean" in metrics_df.columns else None)
                cv_std_col = "CV Std" if "CV Std" in metrics_df.columns else ("cv_std" if "cv_std" in metrics_df.columns else None)
                
                if r2_col and pd.api.types.is_numeric_dtype(metrics_df[r2_col]):
                    fig.add_trace(go.Bar(x=metrics_df.index, y=metrics_df[r2_col], name='R²', showlegend=False), row=1, col=1)
                if rmse_col and pd.api.types.is_numeric_dtype(metrics_df[rmse_col]):
                    fig.add_trace(go.Bar(x=metrics_df.index, y=metrics_df[rmse_col], name='RMSE', showlegend=False), row=1, col=2)
                if mae_col and pd.api.types.is_numeric_dtype(metrics_df[mae_col]):
                    fig.add_trace(go.Bar(x=metrics_df.index, y=metrics_df[mae_col], name='MAE', showlegend=False), row=2, col=1)
                if cv_mean_col and pd.api.types.is_numeric_dtype(metrics_df[cv_mean_col]):
                    error_data = None
                    if cv_std_col and pd.api.types.is_numeric_dtype(metrics_df[cv_std_col]):
                        error_data = dict(type='data', array=metrics_df[cv_std_col])
                    fig.add_trace(go.Bar(x=metrics_df.index, y=metrics_df[cv_mean_col],
                           error_y=error_data,
                           name='CV Score', showlegend=False), row=2, col=2)
                
                fig.update_layout(height=600, title_text="Regression Metrics Overview")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Could not display regression metrics: {str(e)}")
        
        # Prediction vs Actual scatter plots
        with col2:
            try:
                st.subheader("🎯 Predictions vs Actual")
                fig = make_subplots(
                    rows=len(results), cols=1,
                    subplot_titles=[f"{name} Predictions" for name in results.keys()],
                    vertical_spacing=0.1
                )
                
                for i, (model_name, result) in enumerate(results.items(), 1):
                    try:
                        if 'actual' in result and 'predictions' in result:
                            actual = result['actual']
                            predictions = result['predictions']
                            
                            if len(actual) > 0 and len(predictions) > 0:
                                fig.add_trace(
                                    go.Scatter(
                                        x=actual,
                                        y=predictions,
                                        mode='markers',
                                        name=model_name,
                                        showlegend=False
                                    ),
                                    row=i, col=1
                                )
                                
                                # Add perfect prediction line
                                min_val = min(actual.min(), predictions.min())
                                max_val = max(actual.max(), predictions.max())
                                fig.add_trace(
                                    go.Scatter(
                                        x=[min_val, max_val],
                                        y=[min_val, max_val],
                                        mode='lines',
                                        line=dict(dash='dash', color='red'),
                                        name='Perfect Prediction',
                                        showlegend=i==1
                                    ),
                                    row=i, col=1
                                )
                    except Exception as e:
                        print(f"Warning: Could not plot predictions for {model_name}: {str(e)}")
                        continue
                
                fig.update_layout(height=200*len(results), title_text="Prediction Accuracy")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Could not display prediction plots: {str(e)}")
    
    # Model Recommendations
    st.subheader("🏆 Model Recommendations")
    
    if problem_type == "Classification":
        try:
            # Check if required columns exist and provide fallbacks
            available_columns = metrics_df.columns.tolist()
            print(f"Available columns for classification: {available_columns}")
            
            # Check for Accuracy (try both cases)
            accuracy_col = None
            if 'Accuracy' in available_columns:
                accuracy_col = 'Accuracy'
            elif 'accuracy' in available_columns:
                accuracy_col = 'accuracy'
            
            # Check for F1-Score (try both cases)
            f1_col = None
            if 'F1-Score' in available_columns:
                f1_col = 'F1-Score'
            elif 'f1' in available_columns:
                f1_col = 'f1'
            
            # Check for CV Mean (try both cases)
            cv_col = None
            if 'CV Mean' in available_columns:
                cv_col = 'CV Mean'
            elif 'cv_mean' in available_columns:
                cv_col = 'cv_mean'
            
            # Display available metrics
            if accuracy_col and pd.api.types.is_numeric_dtype(metrics_df[accuracy_col]):
                best_accuracy = metrics_df[accuracy_col].idxmax()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Best Accuracy", f"{metrics_df.loc[best_accuracy, accuracy_col]:.3f}", delta=best_accuracy)
                
                if f1_col and pd.api.types.is_numeric_dtype(metrics_df[f1_col]):
                    best_f1 = metrics_df[f1_col].idxmax()
                    with col2:
                        st.metric("Best F1-Score", f"{metrics_df.loc[best_f1, f1_col]:.3f}", delta=best_f1)
                else:
                    with col2:
                        st.metric("Best F1-Score", "N/A", delta="N/A")
                
                if cv_col and pd.api.types.is_numeric_dtype(metrics_df[cv_col]):
                    best_cv = metrics_df[cv_col].idxmax()
                    with col3:
                        st.metric("Best Cross-Validation", f"{metrics_df.loc[best_cv, cv_col]:.3f}", delta=best_cv)
                else:
                    with col3:
                        st.metric("Best Cross-Validation", "N/A", delta="N/A")
                
                # Provide recommendation based on available metrics
                if f1_col and pd.api.types.is_numeric_dtype(metrics_df[f1_col]):
                    st.info(f"💡 **Recommendation**: {best_f1} shows the best overall performance with an F1-score of {metrics_df.loc[best_f1, f1_col]:.3f}")
                elif accuracy_col:
                    st.info(f"💡 **Recommendation**: {best_accuracy} shows the best overall performance with an accuracy of {metrics_df.loc[best_accuracy, accuracy_col]:.3f}")
                else:
                    st.info("💡 **Recommendation**: Check the metrics table above for model performance.")
            else:
                st.warning("⚠️ No accuracy metrics available. Please check your data and model configuration.")
                st.info("Available metrics: " + ", ".join(available_columns))
                
        except Exception as e:
            st.error(f"❌ Error displaying model recommendations: {str(e)}")
            st.info("Please check the metrics table above for available results.")
    
    else:
        try:
            # Check if required columns exist and provide fallbacks
            available_columns = metrics_df.columns.tolist()
            print(f"Available columns for regression: {available_columns}")
            
            # Check for R² (try both cases)
            r2_col = None
            if 'R²' in available_columns:
                r2_col = 'R²'
            elif 'r2' in available_columns:
                r2_col = 'r2'
            
            # Check for RMSE (try both cases)
            rmse_col = None
            if 'RMSE' in available_columns:
                rmse_col = 'RMSE'
            elif 'rmse' in available_columns:
                rmse_col = 'rmse'
            
            # Check for CV Mean (try both cases)
            cv_col = None
            if 'CV Mean' in available_columns:
                cv_col = 'CV Mean'
            elif 'cv_mean' in available_columns:
                cv_col = 'cv_mean'
            
            # Display available metrics
            if r2_col and pd.api.types.is_numeric_dtype(metrics_df[r2_col]):
                best_r2 = metrics_df[r2_col].idxmax()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Best R² Score", f"{metrics_df.loc[best_r2, r2_col]:.3f}", delta=best_r2)
                
                if rmse_col and pd.api.types.is_numeric_dtype(metrics_df[rmse_col]):
                    best_rmse = metrics_df[rmse_col].idxmin()
                    with col2:
                        st.metric("Lowest RMSE", f"{metrics_df.loc[best_rmse, rmse_col]:.3f}", delta=best_rmse)
                else:
                    with col2:
                        st.metric("Lowest RMSE", "N/A", delta="N/A")
                
                if cv_col and pd.api.types.is_numeric_dtype(metrics_df[cv_col]):
                    best_cv = metrics_df[cv_col].idxmax()
                    with col3:
                        st.metric("Best Cross-Validation", f"{metrics_df.loc[best_cv, cv_col]:.3f}", delta=best_cv)
                else:
                    with col3:
                        st.metric("Best Cross-Validation", "N/A", delta="N/A")
                
                # Provide recommendation based on available metrics
                st.info(f"💡 **Recommendation**: {best_r2} shows the best overall performance with the highest R² score of {metrics_df.loc[best_r2, r2_col]:.3f}")
            else:
                st.warning("⚠️ No R² metrics available. Please check your data and model configuration.")
                st.info("Available metrics: " + ", ".join(available_columns))
                
        except Exception as e:
            st.error(f"❌ Error displaying model recommendations: {str(e)}")
            st.info("Please check the metrics table above for available results.")

# Main execution
if __name__ == "__main__":
    show_model_comparison()