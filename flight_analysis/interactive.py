from datetime import datetime
import logging
from flight_analysis import DataAnalyst
from flight_analysis.config import OPENAI_API_KEY

def print_example_questions():
    print("\nExample questions you can ask:")
    print("1. What are our top 5 most profitable routes?")
    print("2. How has booking volume changed month over month?")
    print("3. Which days of the week have the highest cancellation rates?")
    print("4. What's the average delay time by airline?")
    print("5. Which routes have the highest customer satisfaction scores?")
    print("6. What's the correlation between flight delays and customer ratings?")
    print("7. Which airports have the most delayed departures?")
    print("\nType 'examples' to see these questions again")
    print("Type 'quit' to exit")

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize the analyst
    analyst = DataAnalyst()
    
    print("\nWelcome to the Flight Data Analysis System!")
    print("==========================================")
    print_example_questions()
    
    while True:
        try:
            # Get user input
            question = input("\nEnter your business question (or type 'quit' to exit): ").strip()
            
            # Handle special commands
            if question.lower() == 'quit':
                print("Thank you for using the Flight Data Analysis System!")
                break
            elif question.lower() == 'examples':
                print_example_questions()
                continue
            elif not question:
                print("Please enter a question or type 'quit' to exit.")
                continue
                
            # Analyze the question
            print("\nAnalyzing your question... Please wait...")
            results = analyst.analyze(question)
            
            # Print formatted results
            formatted_output = analyst.format_results(results)
            print(formatted_output)
            
            # Ask if user wants to save the results
            save = input("\nWould you like to save these results to a file? (yes/no): ").strip().lower()
            if save.startswith('y'):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"analysis_results_{timestamp}.txt"
                with open(filename, 'w') as f:
                    f.write(formatted_output)
                print(f"Results saved to {filename}")
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            logging.exception("Detailed error information:")
            print("\nPlease try another question or type 'quit' to exit.")

if __name__ == "__main__":
    main()