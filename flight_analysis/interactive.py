import streamlit as st
import logging
from datetime import datetime
from flight_analysis import DataAnalyst
from flight_analysis.config import OPENAI_API_KEY

def setup_quiet_logging():
    """Setup logging without console output"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('processing.log'),
        ]
    )
    
    # Suppress other modules' logging
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('crewai').setLevel(logging.WARNING)
    logging.getLogger('langchain').setLevel(logging.WARNING)

def get_example_questions():
    return [
        "What are our top 5 most profitable routes?",
        "How has booking volume changed month over month?",
        "Which days of the week have the highest cancellation rates?",
        "What's the average delay time by airline?",
        "Which routes have the highest customer satisfaction scores?",
        "What's the correlation between flight delays and customer ratings?",
        "Which airports have the most delayed departures?"
    ]

def main():
    # Setup page config
    st.set_page_config(
        page_title="Flight Data Analysis System",
        page_icon="✈️",
        layout="wide"
    )

    # Setup logging
    setup_quiet_logging()
    
    # Initialize analyst if not in session state
    if 'analyst' not in st.session_state:
        st.session_state.analyst = DataAnalyst()
    
    # Header
    st.title("✈️ Data Analysis System")
    st.markdown("---")

    
    # Main input area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Question input
        question = st.text_area(
            "Enter your business question:",
            value="",
            height=100,
            placeholder="Type your question here..."
        )

    with col2:
        st.markdown("<br>" * 2, unsafe_allow_html=True)  # Space for alignment
        analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
        
        if question and analyze_button:
            with st.spinner('Analyzing your question... Please wait...'):
                try:
                    # Get analysis results
                    results = st.session_state.analyst.analyze(question)
                    formatted_output = st.session_state.analyst.format_results(results)
                    
                    # Store results in session state
                    st.session_state.last_results = formatted_output
                    st.session_state.show_save = True
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    logging.exception("Detailed error information:")
    
    # Display results if available
    if hasattr(st.session_state, 'last_results'):
        st.markdown("### 📊 Analysis Results")
        st.markdown("---")
        
        # Format and display results
        results_text = st.session_state.last_results
        
        # Split into sections
        if "EXECUTIVE SUMMARY" in results_text:
            sections = results_text.split("---")
            for section in sections:
                if "EXECUTIVE SUMMARY" in section:
                    st.info(section.strip())
                elif "KEY FINDINGS" in section:
                    st.success(section.strip())
                elif "BUSINESS IMPLICATIONS" in section:
                    st.warning(section.strip())
                else:
                    st.write(section.strip())
        else:
            st.write(results_text)
        
        # Save results option
        if hasattr(st.session_state, 'show_save'):
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if st.button("💾 Save Results to File", use_container_width=True):
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"analysis_results_{timestamp}.txt"
                    
                    with open(filename, 'w') as f:
                        f.write(st.session_state.last_results)
                    
                    st.success(f"Results saved to {filename}")
                    
    # Footer
    st.markdown("---")

if __name__ == "__main__":
    main()