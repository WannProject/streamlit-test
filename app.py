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
    page_title="Dashboard Analisis & ML",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS =====
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# ===== TITLE =====
st.title("📊 Dashboard Analisis Data & Machine Learning")
st.markdown("---")

# ===== SIDEBAR =====
with st.sidebar:
    st.header("⚙️ Pengaturan")
    
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload Dataset (CSV)", 
        type=['csv'],
        help="Upload file CSV untuk analisis"
    )
    
    st.markdown("---")
    
    # Sample dataset option
    use_sample = st.checkbox("Gunakan Sample Dataset (Iris)", value=True if not uploaded_file else False)
    
    st.markdown("---")
    st.info("💡 **Tips**: Upload dataset Anda atau gunakan sample dataset untuk eksplorasi")

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

# Load dataset
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.success(f"✅ Dataset berhasil diupload: **{len(df)}** baris, **{len(df.columns)}** kolom")
elif use_sample:
    df = load_sample_data()
    st.info("📋 Menggunakan **Iris Dataset** sebagai contoh")
else:
    st.warning("⚠️ Silakan upload dataset atau pilih sample dataset")
    st.stop()

# ===== TABS UTAMA =====
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Overview", 
    "📊 Exploratory Data Analysis", 
    "📈 Visualisasi", 
    "🤖 Machine Learning",
    "📥 Export Data"
])

# ===== TAB 1: OVERVIEW =====
with tab1:
    st.header("📋 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Baris", f"{len(df):,}")
    with col2:
        st.metric("Total Kolom", f"{len(df.columns):,}")
    with col3:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    with col4:
        st.metric("Duplicates", f"{df.duplicated().sum():,}")
    
    st.markdown("---")
    
    # Preview data
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Preview Data")
        n_rows = st.slider("Tampilkan baris:", 5, 50, 10)
        st.dataframe(df.head(n_rows), use_container_width=True)
    
    with col2:
        st.subheader("📊 Tipe Data")
        dtype_df = pd.DataFrame({
            'Kolom': df.dtypes.index,
            'Tipe': df.dtypes.values
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Column info
    st.subheader("📝 Informasi Kolom")
    col_info = pd.DataFrame({
        'Kolom': df.columns,
        'Non-Null Count': df.count().values,
        'Null Count': df.isnull().sum().values,
        'Dtype': df.dtypes.values,
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    st.dataframe(col_info, use_container_width=True, hide_index=True)

# ===== TAB 2: EDA =====
with tab2:
    st.header("📊 Exploratory Data Analysis")
    
    # Statistik Deskriptif
    st.subheader("📈 Statistik Deskriptif")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
        
        st.markdown("---")
        
        # Pilih kolom untuk analisis detail
        st.subheader("🔬 Analisis Detail per Kolom")
        
        selected_col = st.selectbox("Pilih Kolom Numerik:", numeric_cols)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean", f"{df[selected_col].mean():.2f}")
            st.metric("Median", f"{df[selected_col].median():.2f}")
        
        with col2:
            st.metric("Std Dev", f"{df[selected_col].std():.2f}")
            st.metric("Variance", f"{df[selected_col].var():.2f}")
        
        with col3:
            st.metric("Min", f"{df[selected_col].min():.2f}")
            st.metric("Max", f"{df[selected_col].max():.2f}")
        
        with col4:
            st.metric("Q1 (25%)", f"{df[selected_col].quantile(0.25):.2f}")
            st.metric("Q3 (75%)", f"{df[selected_col].quantile(0.75):.2f}")
        
        # Distribution plot
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x=selected_col, nbins=30, 
                             title=f"Distribusi {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, y=selected_col, 
                        title=f"Box Plot {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Correlation Matrix
        if len(numeric_cols) > 1:
            st.subheader("🔗 Correlation Matrix")
            
            corr = df[numeric_cols].corr()
            
            fig = px.imshow(corr, 
                           text_auto=True, 
                           aspect="auto",
                           color_continuous_scale='RdBu_r',
                           title="Heatmap Korelasi")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Tidak ada kolom numerik dalam dataset")

# ===== TAB 3: VISUALISASI =====
with tab3:
    st.header("📈 Visualisasi Data")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    viz_type = st.selectbox(
        "Pilih Tipe Visualisasi:",
        ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", 
         "Box Plot", "Violin Plot", "Pie Chart", "Pair Plot"]
    )
    
    if viz_type == "Scatter Plot":
        if len(numeric_cols) >= 2:
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("X-axis:", numeric_cols)
            with col2:
                y_col = st.selectbox("Y-axis:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            with col3:
                color_col = st.selectbox("Color by:", [None] + categorical_cols + numeric_cols)
            
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                           title=f"{x_col} vs {y_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Butuh minimal 2 kolom numerik")
    
    elif viz_type == "Line Chart":
        if numeric_cols:
            selected_cols = st.multiselect("Pilih Kolom:", numeric_cols, default=numeric_cols[:2])
            if selected_cols:
                fig = px.line(df, y=selected_cols, title="Line Chart")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Tidak ada kolom numerik")
    
    elif viz_type == "Bar Chart":
        if categorical_cols and numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                cat_col = st.selectbox("Kategori:", categorical_cols)
            with col2:
                num_col = st.selectbox("Value:", numeric_cols)
            
            agg_df = df.groupby(cat_col)[num_col].mean().reset_index()
            fig = px.bar(agg_df, x=cat_col, y=num_col,
                        title=f"Average {num_col} by {cat_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Butuh kolom kategorikal dan numerik")
    
    elif viz_type == "Histogram":
        if numeric_cols:
            col = st.selectbox("Pilih Kolom:", numeric_cols)
            bins = st.slider("Jumlah Bins:", 10, 100, 30)
            fig = px.histogram(df, x=col, nbins=bins, title=f"Distribusi {col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Tidak ada kolom numerik")
    
    elif viz_type == "Box Plot":
        if numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                y_col = st.selectbox("Value:", numeric_cols)
            with col2:
                x_col = st.selectbox("Group by:", [None] + categorical_cols)
            
            fig = px.box(df, x=x_col, y=y_col, title=f"Box Plot: {y_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Tidak ada kolom numerik")
    
    elif viz_type == "Violin Plot":
        if numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                y_col = st.selectbox("Value:", numeric_cols)
            with col2:
                x_col = st.selectbox("Group by:", [None] + categorical_cols)
            
            fig = px.violin(df, x=x_col, y=y_col, box=True, title=f"Violin Plot: {y_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Tidak ada kolom numerik")
    
    elif viz_type == "Pie Chart":
        if categorical_cols:
            col = st.selectbox("Pilih Kolom:", categorical_cols)
            value_counts = df[col].value_counts()
            fig = px.pie(values=value_counts.values, names=value_counts.index,
                        title=f"Distribusi {col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Tidak ada kolom kategorikal")
    
    elif viz_type == "Pair Plot":
        if len(numeric_cols) >= 2:
            selected_cols = st.multiselect("Pilih Kolom (max 4):", numeric_cols, 
                                          default=numeric_cols[:min(4, len(numeric_cols))])
            if len(selected_cols) >= 2:
                fig = px.scatter_matrix(df[selected_cols], title="Pair Plot")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Pilih minimal 2 kolom")
        else:
            st.warning("Butuh minimal 2 kolom numerik")

# ===== TAB 4: MACHINE LEARNING =====
with tab4:
    st.header("🤖 Machine Learning Model")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()
    
    if len(numeric_cols) >= 2:
        
        # Pilih target dan features
        st.subheader("1️⃣ Pilih Features dan Target")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_col = st.selectbox("Target Variable:", all_cols)
        
        with col2:
            feature_cols = st.multiselect(
                "Feature Variables:", 
                [col for col in all_cols if col != target_col],
                default=[col for col in numeric_cols if col != target_col][:3]
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
            st.subheader("2️⃣ Konfigurasi Model")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                test_size = st.slider("Test Size:", 0.1, 0.5, 0.2, 0.05)
            with col2:
                random_state = st.number_input("Random State:", 0, 100, 42)
            with col3:
                n_estimators = st.slider("N Estimators:", 10, 200, 100, 10)
            
            if st.button("🚀 Train Model", type="primary"):
                
                with st.spinner("Training model..."):
                    
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
                        
                        st.success("✅ Model berhasil di-training!")
                        
                        st.markdown("---")
                        st.subheader("📊 Model Performance")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accuracy", f"{accuracy:.2%}")
                        with col2:
                            st.metric("Train Size", len(X_train))
                        with col3:
                            st.metric("Test Size", len(X_test))
                        
                        # Confusion Matrix
                        from sklearn.metrics import confusion_matrix
                        cm = confusion_matrix(y_test, y_pred)
                        
                        fig = px.imshow(cm, text_auto=True, 
                                      labels=dict(x="Predicted", y="Actual"),
                                      title="Confusion Matrix")
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
                        
                        st.success("✅ Model berhasil di-training!")
                        
                        st.markdown("---")
                        st.subheader("📊 Model Performance")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("R² Score", f"{r2:.4f}")
                        with col2:
                            st.metric("RMSE", f"{rmse:.4f}")
                        with col3:
                            st.metric("Train Size", len(X_train))
                        with col4:
                            st.metric("Test Size", len(X_test))
                        
                        # Actual vs Predicted
                        fig = px.scatter(x=y_test, y=y_pred,
                                       labels={'x': 'Actual', 'y': 'Predicted'},
                                       title='Actual vs Predicted')
                        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                               y=[y_test.min(), y_test.max()],
                                               mode='lines', name='Perfect Prediction',
                                               line=dict(color='red', dash='dash')))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature Importance
                    st.markdown("---")
                    st.subheader("🎯 Feature Importance")
                    
                    importance_df = pd.DataFrame({
                        'Feature': feature_cols,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(importance_df, x='Importance', y='Feature',
                               orientation='h', title='Feature Importance')
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Pilih target dan minimal 1 feature variable")
    else:
        st.warning("⚠️ Dataset harus memiliki minimal 2 kolom numerik untuk ML")

# ===== TAB 5: EXPORT =====
with tab5:
    st.header("📥 Export Data")
    
    st.subheader("📊 Preview Data yang akan di-export")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💾 Download CSV")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name='analyzed_data.csv',
            mime='text/csv',
        )
    
    with col2:
        st.subheader("📊 Download Excel")
        # Note: pandas Excel writer requires openpyxl
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
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        except ImportError:
            st.warning("⚠️ Install openpyxl untuk export Excel: `pip install openpyxl`")

# ===== FOOTER =====
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Made with ❤️ using Streamlit | © 2025</p>
    </div>
    """,
    unsafe_allow_html=True
)