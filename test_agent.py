from backend.agents.data_agent import DataIntelligenceAgent

from dotenv import load_dotenv
load_dotenv() 

def main():
    """
    Main function to test the DataIntelligenceAgent.
    """
    # Initialize the agent
    data_agent = DataIntelligenceAgent()
    print("Agent initialized.\n")

    # Step 1: Load and clean the data file
    file_path = 'sales_data.csv'
    file_type = 'csv'
    file_name = 'sales_data.csv'
    load_status = data_agent.load_data(file_path, file_type, file_name)
    print(load_status)
    print("The cleaned file has been saved in the /data directory.\n")
    
    # Check the cleaned DataFrame
    print("Cleaned DataFrame Info:")
    data_agent.df.info()
    print("\nCleaned DataFrame Head:")
    print(data_agent.df)
    print(data_agent.df['product'].unique())
    print("-" * 50)

    # Step 2: Test a "total sales" query
    query_total_sales = "List of name of products from los angeles?"
    response_sales = data_agent.handle_query(query_total_sales)
    print(f"Query: '{query_total_sales}'")
    print(f"Response: {response_sales['message']}\n")
    print("-" * 50)
    
    # Step 3: Test a "plot" query
    query_plot = "Plot the product's count over time"
    response_plot = data_agent.handle_query(query_plot)
    print(f"Query: '{query_plot}'")
    print(f"Response: {response_plot['message']}")

    # If the response is an image, save it
    if response_plot["type"] == "image":
        with open("sales_plot.png", "wb") as f:
            f.write(response_plot["data"])
        print("A plot named 'sales_plot.png' has been saved in the project root.")

if __name__ == "__main__":
    main()