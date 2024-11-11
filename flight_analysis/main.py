import logging
from flight_analysis import DataProcessor
from flight_analysis.config import FLIGHT_DATA_PATH, AIRLINE_MAPPING_PATH, OPENAI_API_KEY

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(thread)d - %(filename)s-%(funcName)s:%(lineno)d - %(levelname)s: %(message)s'
    )
    
    # Initialize processor
    processor = DataProcessor(
        input_paths=[FLIGHT_DATA_PATH, AIRLINE_MAPPING_PATH]
    )
    
    try:
        # Process data
        validation_results = processor.process()
        
        # Print summary
        print("\nProcessing Summary:")
        print(f"Total rows processed: {validation_results['total_rows']}")
        print("\nNull counts by column:")
        for column, null_count in validation_results['null_counts'].items():
            percentage = (null_count / validation_results['total_rows']) * 100
            print(f"{column}: {null_count} nulls ({percentage:.2f}%)")
            
    except Exception as e:
        print(f"Error processing data: {str(e)}")

if __name__ == "__main__":
    main()