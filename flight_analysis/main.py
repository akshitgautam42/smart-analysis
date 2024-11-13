import logging
import traceback
from flight_analysis import DataProcessor
from flight_analysis.config import FLIGHT_DATA_PATH, AIRLINE_MAPPING_PATH, OPENAI_API_KEY
from flight_analysis.utils.logging_config import get_processing_logger

def main():
    logger = get_processing_logger()
    logger.info("Starting main processing pipeline")
    
    # Initialize processor
    processor = DataProcessor(
        input_paths=[FLIGHT_DATA_PATH, AIRLINE_MAPPING_PATH]
    )
    
    try:
        # Process data
        validation_results = processor.process()
        
        # Log summary
        logger.info("Processing Summary")
        logger.info(f"Total rows processed: {validation_results['total_rows']}")
        logger.info("Null counts by column:")
        for column, null_count in validation_results['null_counts'].items():
            percentage = (null_count / validation_results['total_rows']) * 100
            logger.info(f"{column}: {null_count} nulls ({percentage:.2f}%)")
            
    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}")
        logger.debug(traceback.format_exc())

if __name__ == "__main__":
    main()