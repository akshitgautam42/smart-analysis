import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
import hashlib
import json
import re
from flight_analysis import DataProcessor, DataAnalyst
from flight_analysis.config import OPENAI_API_KEY

def generate_table_name(file_names):
    """Generate a unique, consistent table name based on input files"""
    combined_names = "_".join(sorted([f.name for f in file_names]))
    hash_object = hashlib.md5(combined_names.encode())
    hash_str = hash_object.hexdigest()[:8]
    
    base_name = file_names[0].name.lower().split('.')[0]
    base_name = ''.join(e for e in base_name if e.isalnum() or e == '_')
    
    return f"{base_name}_{hash_str}"

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def process_uploaded_files(uploaded_files):
    if not uploaded_files:
        return None
        
    file_paths = []
    for uploaded_file in uploaded_files:
        with open(f"temp_{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
            file_paths.append(f"temp_{uploaded_file.name}")
    
    try:
        table_name = generate_table_name(uploaded_files)
        st.session_state.table_name = table_name
        
        processor = DataProcessor(input_paths=file_paths)
        validation_results = processor.process(table_name=table_name)
        st.success("✅ Data processing completed successfully!")
        return validation_results
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        return None

def create_quality_gauge(percentage):
    """Create a gauge chart for data quality score"""
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=100 - percentage,
        title={'text': "Data Quality Score"},
        gauge={
            'axis': {'range': [None, 100]},
            'steps': [
                {'range': [0, 60], 'color': "red"},
                {'range': [60, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))

def display_data_preview(validation_results):
    if not validation_results:
        return

    st.header("📊 Data Validation Results")
    
    # Metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📝 Total Rows", f"{validation_results['total_rows']:,}")
    with col2:
        st.metric("🔢 Total Columns", len(validation_results['null_counts']))
    
    # Calculate overall data quality score
    null_percentages = [(count / validation_results['total_rows']) * 100 
                       for count in validation_results['null_counts'].values()]
    avg_null_percentage = sum(null_percentages) / len(null_percentages)
    with col3:
        st.metric("⭐ Data Quality Score", f"{(100 - avg_null_percentage):.1f}%")
    
    # Create null value analysis DataFrame
    null_df = pd.DataFrame.from_dict(
        validation_results['null_counts'], 
        orient='index', 
        columns=['Null Count']
    )
    null_df['Percentage'] = (null_df['Null Count'] / validation_results['total_rows'] * 100).round(2)
    
    st.subheader("🔍 Data Quality Analysis")
    
    # Create three columns for different visualizations
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create enhanced bar chart
        fig = px.bar(
            null_df,
            y='Percentage',
            title='Missing Data by Column',
            labels={'index': 'Column', 'Percentage': 'Missing Data (%)'},
            height=400,
            color='Percentage',
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(
            showlegend=False,
            title_x=0.5,
            title_font_size=20
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Create gauge chart
        gauge_fig = create_quality_gauge(avg_null_percentage)
        gauge_fig.update_layout(height=400)
        st.plotly_chart(gauge_fig, use_container_width=True)
    
    # Detailed stats in an expandable section
    with st.expander("📋 View Detailed Statistics"):
        st.dataframe(
            null_df.style.background_gradient(subset=['Percentage'], cmap='RdYlGn_r')
                    .format({'Percentage': '{:.2f}%'}),
            height=400
        )

def extract_numbers_from_finding(finding):
    """Extract numbers and entities from finding text"""
    match = re.search(r"(\w+(?:\s+\w+)*)\s+(?:has|with)\s+(\d+)", finding)
    if match:
        return match.groups()
    return None

def create_findings_chart(findings):
    """Create a bar chart for key findings"""
    data = []
    for finding in findings:
        extracted = extract_numbers_from_finding(finding)
        if extracted:
            entity, value = extracted
            data.append({'Entity': entity, 'Value': int(value)})
    
    if data:
        df = pd.DataFrame(data)
        fig = px.bar(
            df,
            x='Entity',
            y='Value',
            title='Key Metrics Visualization',
            color='Value',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(title_x=0.5)
        return fig
    return None

def run_analysis(analyst, question):
    try:
        results = analyst.analyze(question, table_name=st.session_state.table_name)
        
        if 'error' in results:
            st.error(f"❌ Analysis error: {results['error']}")
            return
            
        insights = results['insights']
        
        if isinstance(insights, dict):
            # Executive Summary Section
            st.header("📊 Analysis Results")
            if 'Executive Summary' in insights:
                st.markdown("""
                    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                        <h3>📈 Executive Summary</h3>
                        <p>{}</p>
                    </div>
                """.format(insights['Executive Summary']), unsafe_allow_html=True)
            
            # Key Findings with Visualization
            if 'Key Findings' in insights:
                st.markdown("### 🎯 Key Findings")
                findings_chart = create_findings_chart(insights['Key Findings'])
                if findings_chart:
                    st.plotly_chart(findings_chart, use_container_width=True)
                
                for finding in insights['Key Findings']:
                    st.markdown(f"""
                        <div style='background-color: #e1f5fe; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                            ▪ {finding}
                        </div>
                    """, unsafe_allow_html=True)
            
            # Business Implications and Recommendations
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Business Implications' in insights:
                    st.markdown("### 💡 Business Implications")
                    st.markdown("""
                        <div style='background-color: #fff3e0; padding: 15px; border-radius: 10px;'>
                            {}
                        </div>
                    """.format(insights['Business Implications']), unsafe_allow_html=True)
            
            with col2:
                if 'Actionable Recommendations' in insights:
                    st.markdown("### ✅ Recommendations")
                    for i, rec in enumerate(insights['Actionable Recommendations'], 1):
                        st.markdown(f"""
                            <div style='background-color: #e8f5e9; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                                {i}. {rec}
                            </div>
                        """, unsafe_allow_html=True)
        else:
            st.write(insights)
            
        st.session_state.analysis_results = results
            
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")

def main():
    st.set_page_config(
        page_title="✈️ Flight Data Analyzer",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS styling
    st.markdown("""
        <style>
            .stAlert {
                padding: 20px;
                border-radius: 10px;
            }
            .metric-card {
                background-color: #ffffff;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .st-emotion-cache-1y4p8pa {
                padding: 2rem;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'table_name' not in st.session_state:
        st.session_state.table_name = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Sidebar
    with st.sidebar:
        st.image("https://www.svgrepo.com/show/530440/airplane-2.svg", width=100)
        st.title("📱 Navigation")
        page = st.radio("Select Page", ["📤 Upload Data", "📊 Analysis Dashboard"])
    
    if page == "📤 Upload Data":
        st.title("✈️ Flight Data Upload")
        st.write("Upload your flight booking data files for analysis.")
        
        upload_col1, upload_col2 = st.columns([2, 1])
        with upload_col1:
            uploaded_files = st.file_uploader(
                "Choose CSV files", 
                accept_multiple_files=True,
                type=['csv'],
                help="Upload your flight bookings CSV and airline mapping CSV"
            )
        
        with upload_col2:
            st.markdown("""
                ### 📝 Required Files
                1. Flight bookings data (CSV)
                2. Airline mapping data (CSV)
            """)
        
        if uploaded_files:
            with st.spinner('🔄 Processing data...'):
                validation_results = process_uploaded_files(uploaded_files)
                if validation_results:
                    display_data_preview(validation_results)
                    st.session_state.data_loaded = True
    
    else:  # Analysis Dashboard
        st.title("✈️ Flight Data Analysis Dashboard")
        
        if not st.session_state.data_loaded or not st.session_state.table_name:
            st.warning("⚠️ Please upload data files in the 'Upload Data' page first.")
            return
        
        # Analysis input section
        st.markdown("### 🔍 Ask Questions About Your Flight Data")
        
        # Example questions in an expander
        with st.expander("💡 See Example Questions"):
            st.markdown("""
                - Which airline has the most flights?
                - What are the top 3 most frequented destinations?
                - Show me the average flight delay by airline
                - Which routes have the highest cancellation rates?
                - What is the monthly booking trend?
            """)
        
        question = st.text_area("Enter your business question:", height=100)
        
        if question:
            with st.spinner('🔄 Analyzing data...'):
                analyst = DataAnalyst()
                run_analysis(analyst, question)
        
        # Save results option
        if st.session_state.get('analysis_results'):
            st.divider()
            save_col1, save_col2 = st.columns([1, 4])
            with save_col1:
                if st.button("💾 Save Analysis"):
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"flight_analysis_{timestamp}.json"
                    with open(filename, 'w') as f:
                        json.dump(st.session_state.analysis_results, f, indent=4)
                    st.success(f"✅ Results saved to {filename}")

if __name__ == "__main__":
    setup_logging()
    main()