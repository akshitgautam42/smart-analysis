from pathlib import Path
import streamlit as st
import logging
from datetime import datetime
from flight_analysis import DataAnalyst
from flight_analysis import get_analysis_logger

def setup_quiet_logging():
    """Setup logging for interactive mode"""
    logger = get_analysis_logger()
    logger.info("Initializing interactive mode logging configuration")
    
    # Suppress other modules
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('crewai').setLevel(logging.WARNING)
    logging.getLogger('langchain').setLevel(logging.WARNING)
    
    logger.debug("External module logging levels set to WARNING")
    return logger

def get_results_directory() -> Path:
    """Get the directory for saving results (one level up from chat folder)"""
    logger = logging.getLogger(__name__)
    
    current_dir = Path(__file__).parent  # chat folder
    results_dir = current_dir.parent / "chat" # parent directory
    
    logger.debug(f"Results directory set to: {results_dir}")
    return results_dir

def main():
    logger = setup_quiet_logging()
    logger.info("Starting Flight Data Analysis System")
    
    # Setup page config
    st.set_page_config(
        page_title="Flight Data Analysis System",
        page_icon="✈️",
        layout="wide"
    )
    logger.debug("Page configuration initialized")
    
    # Initialize analyst if not in session state
    if 'analyst' not in st.session_state:
        logger.info("Initializing new DataAnalyst instance")
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
            logger.info(f"Processing analysis request. Question length: {len(question)}")
            with st.spinner('Analyzing your question... Please wait...'):
                try:
                    # Get analysis results
                    logger.debug("Initiating analysis process")
                    results = st.session_state.analyst.analyze(question)
                    formatted_output = st.session_state.analyst.format_results(results)
                    
                    # Store results in session state
                    st.session_state.last_results = formatted_output
                    st.session_state.show_save = True
                    logger.info("Analysis completed successfully")
                except Exception as e:
                    error_msg = f"Error during analysis: {str(e)}"
                    logger.error(error_msg)
                    logger.exception("Detailed error information:")
                    st.error(error_msg)
    
    # Display results if available
    if hasattr(st.session_state, 'last_results'):
        logger.debug("Rendering analysis results")
        st.markdown("### 📊 Analysis Results")
        st.markdown("---")
        
        # Format and display results
        results_text = st.session_state.last_results
        
        # Split into sections
        if "EXECUTIVE SUMMARY" in results_text:
            logger.debug("Rendering formatted sections")
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
            logger.debug("Rendering plain text results")
            st.write(results_text)
        
        # Save results option
        if hasattr(st.session_state, 'show_save'):
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if st.button("💾 Save Results to File", use_container_width=True):
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"analysis_results_{timestamp}.txt"
                    
                    # Save to parent directory
                    save_path = get_results_directory() / filename
                    logger.info(f"Saving analysis results to: {save_path}")
                    
                    try:
                        with open(save_path, 'w') as f:
                            f.write(st.session_state.last_results)
                        
                        success_msg = f"Results saved to {filename}"
                        logger.info(success_msg)
                        st.success(success_msg)
                    except Exception as e:
                        error_msg = f"Error saving results: {str(e)}"
                        logger.error(error_msg)
                        logger.exception("Detailed save error information:")
                        st.error(error_msg)
                    
    # Footer
    st.markdown("---")
    logger.debug("Page rendering completed")

if __name__ == "__main__":
    main()