
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# ===== KONFIGURASI HALAMAN =====
st.set_page_config(
    page_title="Professional Data Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS PROFESIONAL =====
st.markdown("""
    <style>
    /* Global Styles */
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header Styles */
    .main-header {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Card Styles */
    .card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: none;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }
    
    /* Enhanced Streamlit Metrics */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }
    
    div[data-testid="metric-container"] label {
        color: rgba(255, 255, 255, 0.9);
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: white;
        font-weight: 700;
        font-size: 1.8rem;
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .css-1lcbmhc {
        background: rgba(255, 255, 255, 0.95);
    }
    
    /* Tab Styles */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 0.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Dataframe Styles */
    .dataframe {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    
    /* Success/Warning/Info Messages */
    .stSuccess {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        border-radius: 10px;
        padding: 1rem;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);
        border-radius: 10px;
        padding: 1rem;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%);
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Plotly Charts */
    .js-plotly-plot {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    
    /* Selectbox/Slider Styles */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Footer Styles */
    .footer {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-top-color: #667eea;
    }
    </style>
    """, unsafe_allow_html=True)

# ===== HEADER PROFESIONAL =====
st.markdown("""
    <div class="main-header fade-in">
        <h1 style="text-align: center; color: #2d3748; margin-bottom: 0.5rem;">
            🚀 Professional Data Analytics & ML Dashboard
        </h1>
        <p style="text-align: center; color: #718096; font-size: 1.1rem;">
            Advanced Data Analysis, Visualization & Machine Learning Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

# ===== SIDEBAR ENHANCED =====
with st.sidebar:
    st.markdown("""
        <div class="card">
            <h2 style="color: #2d3748; margin-bottom: 1rem;">⚙️ Configuration Panel</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Upload file with enhanced styling
    uploaded_file = st.file_uploader(
        "📁 Upload Dataset (CSV)", 
        type=['csv'],
        help="Upload your CSV file for comprehensive analysis",
        label_visibility="visible"
    )
    
    st.markdown("---")
    
    # Sample dataset option
    use_sample = st.checkbox(
        "🎯 Use Sample Dataset (Iris)", 
        value=True if not uploaded_file else False,
        help="Load the famous Iris dataset for demonstration"
    )
    
    st.markdown("---")
    
    # Enhanced tips section
    st.markdown("""
        <div class="card">
            <h4 style="color: #2d3748; margin-bottom: 0.5rem;">💡 Pro Tips</h4>
            <ul style="color: #4a5568; font-size: 0.9rem; margin: 0;">
                <li>Upload your own dataset or use sample data</li>
                <li>Explore multiple visualization options</li>
                <li>Build ML models with one click</li>
                <li>Export results in various formats</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ===== LOAD DATA =====
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

@st.cache_data
def load_sample_data():
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    return df

# Load dataset dengan enhanced notifications
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.markdown("""
        <div class="card fade-in">
            <h3 style="color: #2d3748; margin-bottom: 0.5rem;">✅ Dataset Loaded Successfully!</h3>
            <p style="color: #4a5568; margin: 0;">
                <strong>{:,}</strong> rows • <strong>{:,}</strong> columns
            </p>
        </div>
        """.format(len(df), len(df.columns)), unsafe_allow_html=True)
elif use_sample:
    df = load_sample_data()
    st.markdown("""
        <div class="card fade-in">
            <h3 style="color: #2d3748; margin-bottom: 0.5rem;">📋 Sample Dataset Loaded</h3>
            <p style="color: #4a5568; margin: 0;">
                Using the famous Iris dataset for demonstration
            </p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div class="card">
            <h3 style="color: #f56565; margin-bottom: 0.5rem;">⚠️ No Dataset Selected</h3>
            <p style="color: #4a5568; margin: 0;">
                Please upload a dataset or use the sample dataset to continue
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ===== TABS UTAMA ENHANCED =====
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard Overview", 
    "🔍 Advanced Analytics", 
    "📈 Interactive Visualizations", 
    "🤖 ML Studio",
    "📥 Export & Share"
])

# ===== TAB 1: OVERVIEW ENHANCED =====
with tab1:
    st.markdown("""
        <div class="card fade-in">
            <h2 style="color: #2d3748; margin-bottom: 1rem;">📊 Dataset Overview</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced metrics with better styling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Total Rows</h4>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.8rem; font-weight: bold;">{len(df):,}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Total Columns</h4>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.8rem; font-weight: bold;">{len(df.columns):,}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Missing Values</h4>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.8rem; font-weight: bold;">{df.isnull().sum().sum():,}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Duplicates</h4>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.8rem; font-weight: bold;">{df.duplicated().sum():,}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced data preview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
            <div class="card">
                <h3 style="color: #2d3748; margin-bottom: 1rem;">🔍 Data Preview</h3>
            </div>
            """, unsafe_allow_html=True)
        n_rows = st.slider("Show rows:", 5, 50, 10, help="Select number of rows to display")
        st.dataframe(df.head(n_rows), use_container_width=True, height=400)
    
    with col2:
        st.markdown("""
            <div class="card">
                <h3 style="color: #2d3748; margin-bottom: 1rem;">📋 Data Types</h3>
            </div>
            """, unsafe_allow_html=True)
        dtype_df = pd.DataFrame({
            'Column': df.dtypes.index,
            'Type': df.dtypes.values
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True, height=400)
    
    st.markdown("---")
    
    # Enhanced column information
    st.markdown("""
        <div class="card">
            <h3 style="color: #2d3748; margin-bottom: 1rem;">📝 Detailed Column Information</h3>
        </div>
        """, unsafe_allow_html=True)
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Non-Null Count': df.count().values,
        'Null Count': df.isnull().sum().values,
        'Data Type': df.dtypes.values,
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    st.dataframe(col_info, use_container_width=True, hide_index=True, height=300)

# ===== TAB 2: ADVANCED ANALYTICS =====
with tab2:
    st.markdown("""
        <div class="card fade-in">
            <h2 style="color: #2d3748; margin-bottom: 1rem;">🔍 Advanced Analytics</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced statistics section
    st.markdown("""
        <div class="card">
            <h3 style="color: #2d3748; margin-bottom: 1rem;">📈 Descriptive Statistics</h3>
        </div>
        """, unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        st.dataframe(df[numeric_cols].describe().T, use_container_width=True, height=300)
        
        st.markdown("---")
        
        # Enhanced column analysis
        st.markdown("""
            <div class="card">
                <h3 style="color: #2d3748; margin-bottom: 1rem;">🔬 Detailed Column Analysis</h3>
            </div>
            """, unsafe_allow_html=True)
        
        selected_col = st.selectbox("Select Numeric Column:", numeric_cols, help="Choose a column for detailed analysis")
        
        # Enhanced metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0; font-size: 0.8rem; opacity: 0.9;">Mean</h4>
                    <p style="margin: 0.3rem 0 0 0; font-size: 1.4rem; font-weight: bold;">{df[selected_col].mean():.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown(f"""
                <div class="metric-card" style="margin-top: 0.5rem;">
                    <h4 style="margin: 0; font-size: 0.8rem; opacity: 0.9;">Median</h4>
                    <p style="margin: 0.3rem 0 0 0; font-size: 1.4rem; font-weight: bold;">{df[selected_col].median():.2f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0; font-size: 0.8rem; opacity: 0.9;">Std Dev</h4>
                    <p style="margin: 0.3rem 0 0 0; font-size: 1.4rem; font-weight: bold;">{df[selected_col].std():.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown(f"""
                <div class="metric-card" style="margin-top: 0.5rem;">
                    <h4 style="margin: 0; font-size: 0.8rem; opacity: 0.9;">Variance</h4>
                    <p style="margin: 0.3rem 0 0 0; font-size: 1.4rem; font-weight: bold;">{df[selected_col].var():.2f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0; font-size: 0.8rem; opacity: 0.9;">Min</h4>
                    <p style="margin: 0.3rem 0 0 0; font-size: 1.4rem; font-weight: bold;">{df[selected_col].min():.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown(f"""
                <div class="metric-card" style="margin-top: 0.5rem;">
                    <h4 style="margin: 0; font-size: 0.8rem; opacity: 0.9;">Max</h4>
                    <p style="margin: 0.3rem 0 0 0; font-size: 1.4rem; font-weight: bold;">{df[selected_col].max():.2f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0; font-size: 0.8rem; opacity: 0.9;">Q1 (25%)</h4>
                    <p style="margin: 0.3rem 0 0 0; font-size: 1.4rem; font-weight: bold;">{df[selected_col].quantile(0.25):.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown(f"""
                <div class="metric-card" style="margin-top: 0.5rem;">
                    <h4 style="margin: 0; font-size: 0.8rem; opacity: 0.9;">Q3 (75%)</h4>
                    <p style="margin: 0.3rem 0 0 0; font-size: 1.4rem; font-weight: bold;">{df[selected_col].quantile(0.75):.2f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Enhanced distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class="card">
                    <h4 style="color: #2d3748; margin-bottom: 1rem;">📊 Distribution</h4>
                </div>
                """, unsafe_allow_html=True)
            fig = px.histogram(df, x=selected_col, nbins=30, 
                             title=f"Distribution of {selected_col}",
                             color_discrete_sequence=['#667eea'])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
                <div class="card">
                    <h4 style="color: #2d3748; margin-bottom: 1rem;">📊 Box Plot</h4>
                </div>
                """, unsafe_allow_html=True)
            fig = px.box(df, y=selected_col, 
                        title=f"Box Plot of {selected_col}",
                        color_discrete_sequence=['#764ba2'])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Enhanced correlation matrix
        if len(numeric_cols) > 1:
            st.markdown("""
                <div class="card">
                    <h3 style="color: #2d3748; margin-bottom: 1rem;">🔗 Correlation Matrix</h3>
                </div>
                """, unsafe_allow_html=True)
            
            corr = df[numeric_cols].corr()
            
            fig = px.imshow(corr, 
                           text_auto=True, 
                           aspect="auto",
                           color_continuous_scale='RdBu_r',
                           title="Feature Correlation Heatmap")
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("""
            <div class="card">
                <h3 style="color: #f56565; margin-bottom: 0.5rem;">⚠️ No Numeric Columns</h3>
                <p style="color: #4a5568; margin: 0;">
                    The dataset doesn't contain any numeric columns for analysis
                </p>
            </div>
            """, unsafe_allow_html=True)

# ===== TAB 3: INTERACTIVE VISUALIZATIONS =====
with tab3:
    st.markdown("""
        <div class="card fade-in">
            <h2 style="color: #2d3748; margin-bottom: 1rem;">📈 Interactive Visualizations</h2>
        </div>
        """, unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    viz_type = st.selectbox(
        "🎨 Select Visualization Type:",
        ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", 
         "Box Plot", "Violin Plot", "Pie Chart", "Pair Plot"],
        help="Choose the type of visualization you want to create"
    )
    
    st.markdown("---")
    
    if viz_type == "Scatter Plot":
        if len(numeric_cols) >= 2:
            st.markdown("""
                <div class="card">
                    <h3 style="color: #2d3748; margin-bottom: 1rem;">📊 Scatter Plot Configuration</h3>
                </div>
                """, unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("X-axis:", numeric_cols)
            with col2:
                y_col = st.selectbox("Y-axis:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            with col3:
                color_col = st.selectbox("Color by:", [None] + categorical_cols + numeric_cols)
            
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                           title=f"{x_col} vs {y_col}",
                           color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
                <div class="card">
                    <h3 style="color: #f56565; margin-bottom: 0.5rem;">⚠️ Insufficient Columns</h3>
                    <p style="color: #4a5568; margin: 0;">
                        Scatter plot requires at least 2 numeric columns
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    elif viz_type == "Line Chart":
        if numeric_cols:
            st.markdown("""
                <div class="card">
                    <h3 style="color: #2d3748; margin-bottom: 1rem;">📈 Line Chart Configuration</h3>
                </div>
                """, unsafe_allow_html=True)
            selected_cols = st.multiselect("Select Columns:", numeric_cols, default=numeric_cols[:2])
            if selected_cols:
                fig = px.line(df, y=selected_cols, title="Multi-Line Chart",
                            color_discrete_sequence=px.colors.qualitative.Set3)
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
                <div class="card">
                    <h3 style="color: #f56565; margin-bottom: 0.5rem;">⚠️ No Numeric Columns</h3>
                    <p style="color: #4a5568; margin: 0;">
                        Line chart requires numeric columns
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    elif viz_type == "Bar Chart":
        if categorical_cols and numeric_cols:
            st.markdown("""
                <div class="card">
                    <h3 style="color: #2d3748; margin-bottom: 1rem;">📊 Bar Chart Configuration</h3>
                </div>
                """, unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                cat_col = st.selectbox("Category:", categorical_cols)
            with col2:
                num_col = st.selectbox("Value:", numeric_cols)
            
            agg_df = df.groupby(cat_col)[num_col].mean().reset_index()
            fig = px.bar(agg_df, x=cat_col, y=num_col,
                        title=f"Average {num_col} by {cat_col}",
                        color_discrete_sequence=['#667eea'])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
                <div class="card">
                    <h3 style="color: #f56565; margin-bottom: 0.5rem;">⚠️ Missing Columns</h3>
                    <p style="color: #4a5568; margin: 0;">
                        Bar chart requires both categorical and numeric columns
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    elif viz_type == "Histogram":
        if numeric_cols:
            st.markdown("""
                <div class="card">
                    <h3 style="color: #2d3748; margin-bottom: 1rem;">📊 Histogram Configuration</h3>
                </div>
                """, unsafe_allow_html=True)
            col = st.selectbox("Select Column:", numeric_cols)
            bins = st.slider("Number of Bins:", 10, 100, 30)
            fig = px.histogram(df, x=col, nbins=bins, title=f"Distribution of {col}",
                             color_discrete_sequence=['#764ba2'])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
                <div class="card">
                    <h3 style="color: #f56565; margin-bottom: 0.5rem;">⚠️ No Numeric Columns</h3>
                    <p style="color: #4a5568; margin: 0;">
                        Histogram requires numeric columns
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    elif viz_type == "Box Plot":
        if numeric_cols:
            st.markdown("""
                <div class="card">
                    <h3 style="color: #2d3748; margin-bottom: 1rem;">📊 Box Plot Configuration</h3>
                </div>
                """, unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                y_col = st.selectbox("Value:", numeric_cols)
            with col2:
                x_col = st.selectbox("Group by:", [None] + categorical_cols)
            
            fig = px.box(df, x=x_col, y=y_col, title=f"Box Plot: {y_col}",
                        color_discrete_sequence=['#667eea'])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
                <div class="card">
                    <h3 style="color: #f56565; margin-bottom: 0.5rem;">⚠️ No Numeric Columns</h3>
                    <p style="color: #4a5568; margin: 0;">
                        Box plot requires numeric columns
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    elif viz_type == "Violin Plot":
        if numeric_cols:
            st.markdown("""
                <div class="card">
                    <h3 style="color: #2d3748; margin-bottom: 1rem;">🎻 Violin Plot Configuration</h3>
                </div>
                """, unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                y_col = st.selectbox("Value:", numeric_cols)
            with col2:
                x_col = st.selectbox("Group by:", [None] + categorical_cols)
            
            fig = px.violin(df, x=x_col, y=y_col, box=True, title=f"Violin Plot: {y_col}",
                          color_discrete_sequence=['#764ba2'])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
                <div class="card">
                    <h3 style="color: #f56565; margin-bottom: 0.5rem;">⚠️ No Numeric Columns</h3>
                    <p style="color: #4a5568; margin: 0;">
                        Violin plot requires numeric columns
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    elif viz_type == "Pie Chart":
        if categorical_cols:
            st.markdown("""
                <div class="card">
                    <h3 style="color: #2d3748; margin-bottom: 1rem;">🥧 Pie Chart Configuration</h3>
                </div>
                """, unsafe_allow_html=True)
            col = st.selectbox("Select Column:", categorical_cols)
            value_counts = df[col].value_counts()
            fig = px.pie(values=value_counts.values, names=value_counts.index,
                        title=f"Distribution of {col}",
                        color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
                <div class="card">
                    <h3 style="color: #f56565; margin-bottom: 0.5rem;">⚠️ No Categorical Columns</h3>
                    <p style="color: #4a5568; margin: 0;">
                        Pie chart requires categorical columns
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    elif viz_type == "Pair Plot":
        if len(numeric_cols) >= 2:
            st.markdown("""
                <div class="card">
                    <h3 style="color: #2d3748; margin-bottom: 1rem;">🔗 Pair Plot Configuration</h3>
                </div>
                """, unsafe_allow_html=True)
            selected_cols = st.multiselect("Select Columns (max 4):", numeric_cols, 
                                          default=numeric_cols[:min(4, len(numeric_cols))])
            if len(selected_cols) >= 2:
                fig = px.scatter_matrix(df[selected_cols], title="Pair Plot Matrix",
                                       color_discrete_sequence=px.colors.qualitative.Set3)
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("""
                    <div class="card">
                        <h3 style="color: #f56565; margin-bottom: 0.5rem;">⚠️ Insufficient Columns</h3>
                        <p style="color: #4a5568; margin: 0;">
                            Please select at least 2 columns for pair plot
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="card">
                    <h3 style="color: #f56565; margin-bottom: 0.5rem;">⚠️ Insufficient Columns</h3>
                    <p style="color: #4a5568; margin: 0;">
                        Pair plot requires at least 2 numeric columns
                    </p>
                </div>
                """, unsafe_allow_html=True)

# ===== TAB 4: ML STUDIO ENHANCED =====
with tab4:
    st.markdown("""
        <div class="card fade-in">
            <h2 style="color: #2d3748; margin-bottom: 1rem;">🤖 Machine Learning Studio</h2>
        </div>
        """, unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()
    
    if len(numeric_cols) >= 2:
        
        # Enhanced feature selection
        st.markdown("""
            <div class="card">
                <h3 style="color: #2d3748; margin-bottom: 1rem;">1️⃣ Feature & Target Selection</h3>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_col = st.selectbox("🎯 Target Variable:", all_cols, help="Select the variable you want to predict")
        
        with col2:
            feature_cols = st.multiselect(
                "📊 Feature Variables:",
                [col for col in all_cols if col != target_col],
                default=[col for col in numeric_cols if col != target_col][:3],
                help="Select the features to use for prediction"
            )
        
        if target_col and feature_cols:
            
            # Prepare data
            X = df[feature_cols].copy()
            y = df[target_col].copy()
            
            # Handle categorical features
            le_dict = {}
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                le_dict[col] = le
            
            # Check if classification or regression
            is_classification = y.dtype == 'object' or y.nunique() < 10
            
            if is_classification and y.dtype == 'object':
                le_target = LabelEncoder()
                y = le_target.fit_transform(y.astype(str))
            
            st.markdown("---")
            
            # Enhanced model configuration
            st.markdown("""
                <div class="card">
                    <h3 style="color: #2d3748; margin-bottom: 1rem;">2️⃣ Model Configuration</h3>
                </div>
                """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                test_size = st.slider("📊 Test Size:", 0.1, 0.5, 0.2, 0.05, help="Proportion of data for testing")
            with col2:
                random_state = st.number_input("🎲 Random State:", 0, 100, 42, help="Seed for reproducibility")
            with col3:
                n_estimators = st.slider("🌳 N Estimators:", 10, 200, 100, 10, help="Number of trees in the forest")
            
            if st.button("🚀 Train Model", type="primary", help="Click to train the ML model"):
                
                with st.spinner("🤖 Training model... This may take a moment..."):
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    
                    # Train model
                    if is_classification:
                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            random_state=random_state
                        )
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        st.markdown("""
                            <div class="card fade-in">
                                <h3 style="color: #2d3748; margin-bottom: 1rem;">✅ Model Training Complete!</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Enhanced performance metrics
                        st.markdown("""
                            <div class="card">
                                <h3 style="color: #2d3748; margin-bottom: 1rem;">📊 Model Performance</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"""
                                <div class="metric-card">
                                    <h4 style="margin: 0; font-size: 0.8rem; opacity: 0.9;">Accuracy</h4>
                                    <p style="margin: 0.3rem 0 0 0; font-size: 1.6rem; font-weight: bold;">{accuracy:.2%}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"""
                                <div class="metric-card">
                                    <h4 style="margin: 0; font-size: 0.8rem; opacity: 0.9;">Train Size</h4>
                                    <p style="margin: 0.3rem 0 0 0; font-size: 1.6rem; font-weight: bold;">{len(X_train)}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        with col3:
                            st.markdown(f"""
                                <div class="metric-card">
                                    <h4 style="margin: 0; font-size: 0.8rem; opacity: 0.9;">Test Size</h4>
                                    <p style="margin: 0.3rem 0 0 0; font-size: 1.6rem; font-weight: bold;">{len(X_test)}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Enhanced confusion matrix
                        st.markdown("---")
                        st.markdown("""
                            <div class="card">
                                <h3 style="color: #2d3748; margin-bottom: 1rem;">🎯 Confusion Matrix</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        from sklearn.metrics import confusion_matrix
                        cm = confusion_matrix(y_test, y_pred)
                        
                        fig = px.imshow(cm, text_auto=True,
                                      labels=dict(x="Predicted", y="Actual"),
                                      title="Confusion Matrix",
                                      color_continuous_scale='Blues')
                        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        model = RandomForestRegressor(
                            n_estimators=n_estimators,
                            random_state=random_state
                        )
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, y_pred)
                        
                        st.markdown("""
                            <div class="card fade-in">
                                <h3 style="color: #2d3748; margin-bottom: 1rem;">✅ Model Training Complete!</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Enhanced performance metrics
                        st.markdown("""
                            <div class="card">
                                <h3 style="color: #2d3748; margin-bottom: 1rem;">📊 Model Performance</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(f"""
                                <div class="metric-card">
                                    <h4 style="margin: 0; font-size: 0.8rem; opacity: 0.9;">R² Score</h4>
                                    <p style="margin: 0.3rem 0 0 0; font-size: 1.4rem; font-weight: bold;">{r2:.4f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"""
                                <div class="metric-card">
                                    <h4 style="margin: 0; font-size: 0.8rem; opacity: 0.9;">RMSE</h4>
                                    <p style="margin: 0.3rem 0 0 0; font-size: 1.4rem; font-weight: bold;">{rmse:.4f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        with col3:
                            st.markdown(f"""
                                <div class="metric-card">
                                    <h4 style="margin: 0; font-size: 0.8rem; opacity: 0.9;">Train Size</h4>
                                    <p style="margin: 0.3rem 0 0 0; font-size: 1.4rem; font-weight: bold;">{len(X_train)}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        with col4:
                            st.markdown(f"""
                                <div class="metric-card">
                                    <h4 style="margin: 0; font-size: 0.8rem; opacity: 0.9;">Test Size</h4>
                                    <p style="margin: 0.3rem 0 0 0; font-size: 1.4rem; font-weight: bold;">{len(X_test)}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Enhanced actual vs predicted plot
                        st.markdown("---")
                        st.markdown("""
                            <div class="card">
                                <h3 style="color: #2d3748; margin-bottom: 1rem;">📈 Actual vs Predicted</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        fig = px.scatter(x=y_test, y=y_pred,
                                       labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                                       title='Actual vs Predicted Comparison')
                        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                               y=[y_test.min(), y_test.max()],
                                               mode='lines', name='Perfect Prediction',
                                               line=dict(color='red', dash='dash')))
                        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Enhanced feature importance
                    st.markdown("---")
                    st.markdown("""
                        <div class="card">
                            <h3 style="color: #2d3748; margin-bottom: 1rem;">🎯 Feature Importance</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    importance_df = pd.DataFrame({
                        'Feature': feature_cols,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(importance_df, x='Importance', y='Feature',
                               orientation='h', title='Feature Importance Ranking',
                               color_discrete_sequence=['#667eea'])
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
                <div class="card">
                    <h3 style="color: #f56565; margin-bottom: 0.5rem;">⚠️ Missing Selection</h3>
                    <p style="color: #4a5568; margin: 0;">
                        Please select target and at least 1 feature variable
                    </p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="card">
                <h3 style="color: #f56565; margin-bottom: 0.5rem;">⚠️ Insufficient Data</h3>
                <p style="color: #4a5568; margin: 0;">
                    Dataset must have at least 2 numeric columns for machine learning
                </p>
            </div>
            """, unsafe_allow_html=True)

# ===== TAB 5: EXPORT & SHARE ENHANCED =====
with tab5:
    st.markdown("""
        <div class="card fade-in">
            <h2 style="color: #2d3748; margin-bottom: 1rem;">📥 Export & Share Results</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced preview section
    st.markdown("""
        <div class="card">
            <h3 style="color: #2d3748; margin-bottom: 1rem;">📊 Data Preview</h3>
            <p style="color: #4a5568; margin: 0;">
                Preview of the data that will be exported
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True, height=300)
    
    st.markdown("---")
    
    # Enhanced export options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="card">
                <h3 style="color: #2d3748; margin-bottom: 1rem;">💾 CSV Export</h3>
                <p style="color: #4a5568; margin: 0;">
                    Download data in CSV format
                </p>
            </div>
            """, unsafe_allow_html=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name='analyzed_data.csv',
            mime='text/csv',
            help="Download your analyzed data as a CSV file"
        )
    
    with col2:
        st.markdown("""
            <div class="card">
                <h3 style="color: #2d3748; margin-bottom: 1rem;">📊 Excel Export</h3>
                <p style="color: #4a5568; margin: 0;">
                    Download data in Excel format
                </p>
            </div>
            """, unsafe_allow_html=True)
        try:
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            excel_data = output.getvalue()
            
            st.download_button(
                label="📥 Download as Excel",
                data=excel_data,
                file_name='analyzed_data.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                help="Download your analyzed data as an Excel file"
            )
        except ImportError:
            st.markdown("""
                <div class="card">
                    <h3 style="color: #f56565; margin-bottom: 0.5rem;">⚠️ Missing Dependency</h3>
                    <p style="color: #4a5568; margin: 0;">
                        Install openpyxl for Excel export: <code>pip install openpyxl</code>
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced statistics export
    st.markdown("""
        <div class="card">
            <h3 style="color: #2d3748; margin-bottom: 1rem;">📈 Export Statistics</h3>
            <p style="color: #4a5568; margin: 0;">
                Download summary statistics of your data
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        stats_df = df[numeric_cols].describe().T
        stats_csv = stats_df.to_csv().encode('utf-8')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                label="📊 Download Statistics CSV",
                data=stats_csv,
                file_name='data_statistics.csv',
                mime='text/csv',
                help="Download descriptive statistics"
            )
        
        with col2:
            # Create correlation matrix for export
            corr_df = df[numeric_cols].corr()
            corr_csv = corr_df.to_csv().encode('utf-8')
            st.download_button(
                label="🔗 Download Correlation Matrix",
                data=corr_csv,
                file_name='correlation_matrix.csv',
                mime='text/csv',
                help="Download correlation matrix"
            )
        
        with col3:
            # Create data summary
            summary_data = {
                'Metric': ['Total Rows', 'Total Columns', 'Numeric Columns', 'Categorical Columns',
                          'Missing Values', 'Duplicates', 'Memory Usage (MB)'],
                'Value': [len(df), len(df.columns), len(numeric_cols),
                         len(df.select_dtypes(include=['object']).columns),
                         df.isnull().sum().sum(), df.duplicated().sum(),
                         round(df.memory_usage(deep=True).sum() / 1024**2, 2)]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_csv = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📋 Download Data Summary",
                data=summary_csv,
                file_name='data_summary.csv',
                mime='text/csv',
                help="Download data summary information"
            )

# ===== PROFESSIONAL FOOTER =====
st.markdown("---")
st.markdown("""
    <div class="footer fade-in">
        <h3 style="color: #2d3748; margin-bottom: 1rem;">🚀 Professional Data Analytics Dashboard</h3>
        <p style="color: #4a5568; margin-bottom: 1rem;">
            Advanced analytics platform powered by modern data science technologies
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 1rem;">
            <span style="color: #667eea;">📊 Data Analysis</span>
            <span style="color: #764ba2;">🤖 Machine Learning</span>
            <span style="color: #667eea;">📈 Visualizations</span>
            <span style="color: #764ba2;">📥 Export Options</span>
        </div>
        <p style="color: #718096; font-size: 0.9rem; margin: 0;">
            Built with ❤️ using Streamlit • Powered by wanndev • © 2025
        </p>
        <p style="color: #a0aec0; font-size: 0.8rem; margin: 0.5rem 0 0 0;">
            Version 2.0 • Professional Edition
        </p>
    </div>
    """, unsafe_allow_html=True)