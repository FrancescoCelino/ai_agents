from langchain_core.tools import tool
import os
from dotenv import load_dotenv
from langsmith import utils
import llm.tools
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
print(openai_api_key)

print(f"LangSmith tracing is enabled: {utils.tracing_is_enabled()}")

import sqlite3
import requests
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

def get_engine_for_chinook_db():
    """
    Pull SQL file, populate in-memory database, and create engine.
    
    Downloads the Chinook database SQL script from GitHub and creates an in-memory 
    SQLite database populated with the sample data.
    
    Returns:
        sqlalchemy.engine.Engine: SQLAlchemy engine connected to the in-memory database
    """
    # Download the Chinook database SQL script from the official repository
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(url)
    sql_script = response.text

    # Create an in-memory SQLite database connection
    # check_same_thread=False allows the connection to be used across threads
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    
    # Execute the SQL script to populate the database with sample data
    connection.executescript(sql_script)
    
    # Create and return a SQLAlchemy engine that uses the populated connection
    return create_engine(
        "sqlite://",  # SQLite URL scheme
        creator=lambda: connection,  # Function that returns the database connection
        poolclass=StaticPool,  # Use StaticPool to maintain single connection
        connect_args={"check_same_thread": False},  # Allow cross-thread usage
    )

# Initialize the database engine with the Chinook sample data
engine = get_engine_for_chinook_db()

# Create a LangChain SQLDatabase wrapper around the engine
# This provides convenient methods for database operations and query execution
db = SQLDatabase(engine)

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# Initialize long-term memory store for persistent data between conversations
in_memory_store = InMemoryStore()

# Initialize checkpointer for short-term memory within a single thread/conversation
checkpointer = MemorySaver()

from typing_extensions import TypedDict
from typing import Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages.utils import AnyMessage
from langgraph.managed.is_last_step import RemainingSteps

class State(TypedDict):
    """
    State schema for the multi-agent customer support workflow.
    
    This defines the shared data structure that flows between nodes in the graph,
    representing the current snapshot of the conversation and agent state.
    """
    # Customer identifier retrieved from account verification
    customer_id: str
    
    # Conversation history with automatic message aggregation
    messages: Annotated[list[AnyMessage], add_messages]
    
    # User preferences and context loaded from long-term memory store
    loaded_memory: str
    
    # Counter to prevent infinite recursion in agent workflow
    remaining_steps: RemainingSteps 

from langchain_core.tools import tool
import ast

@tool 
def get_invoices_by_customer_sorted_by_date(customer_id: str) -> list[dict]:
    """
    Look up all invoices for a customer using their ID.
    The invoices are sorted in descending order by invoice date, which helps when the customer wants to view their most recent/oldest invoice, or if 
    they want to view invoices within a specific date range.
    
    Args:
        customer_id (str): customer_id, which serves as the identifier.
    
    Returns:
        list[dict]: A list of invoices for the customer.
    """
    return db.run(f"SELECT * FROM Invoice WHERE CustomerId = {customer_id} ORDER BY InvoiceDate DESC;")


@tool 
def get_invoices_sorted_by_unit_price(customer_id: str) -> list[dict]:
    """
    Use this tool when the customer wants to know the details of one of their invoices based on the unit price/cost of the invoice.
    This tool looks up all invoices for a customer, and sorts the unit price from highest to lowest. In order to find the invoice associated with the customer, 
    we need to know the customer ID.
    
    Args:
        customer_id (str): customer_id, which serves as the identifier.
    
    Returns:
        list[dict]: A list of invoices sorted by unit price.
    """
    query = f"""
        SELECT Invoice.*, InvoiceLine.UnitPrice
        FROM Invoice
        JOIN InvoiceLine ON Invoice.InvoiceId = InvoiceLine.InvoiceId
        WHERE Invoice.CustomerId = {customer_id}
        ORDER BY InvoiceLine.UnitPrice DESC;
    """
    return db.run(query)


@tool
def get_employee_by_invoice_and_customer(invoice_id: str, customer_id: str) -> dict:
    """
    This tool will take in an invoice ID and a customer ID and return the employee information associated with the invoice.

    Args:
        invoice_id (int): The ID of the specific invoice.
        customer_id (str): customer_id, which serves as the identifier.

    Returns:
        dict: Information about the employee associated with the invoice.
    """

    query = f"""
        SELECT Employee.FirstName, Employee.Title, Employee.Email
        FROM Employee
        JOIN Customer ON Customer.SupportRepId = Employee.EmployeeId
        JOIN Invoice ON Invoice.CustomerId = Customer.CustomerId
        WHERE Invoice.InvoiceId = ({invoice_id}) AND Invoice.CustomerId = ({customer_id});
    """
    
    employee_info = db.run(query, include_columns=True)
    
    if not employee_info:
        return f"No employee found for invoice ID {invoice_id} and customer identifier {customer_id}."
    return employee_info

# Create a list of all invoice-related tools for the agent
invoice_tools = [get_invoices_by_customer_sorted_by_date, get_invoices_sorted_by_unit_price, get_employee_by_invoice_and_customer]

invoice_subagent_prompt = """
    You are a subagent among a team of assistants. You are specialized for retrieving and processing invoice information. You are routed for invoice-related portion of the questions, so only respond to them.. 

    You have access to three tools. These tools enable you to retrieve and process invoice information from the database. Here are the tools:
    - get_invoices_by_customer_sorted_by_date: This tool retrieves all invoices for a customer, sorted by invoice date.
    - get_invoices_sorted_by_unit_price: This tool retrieves all invoices for a customer, sorted by unit price.
    - get_employee_by_invoice_and_customer: This tool retrieves the employee information associated with an invoice and a customer.
    
    If you are unable to retrieve the invoice information, inform the customer you are unable to retrieve the information, and ask if they would like to search for something else.
    
    CORE RESPONSIBILITIES:
    - Retrieve and process invoice information from the database
    - Provide detailed information about invoices, including customer details, invoice dates, total amounts, employees associated with the invoice, etc. when the customer asks for it.
    - Always maintain a professional, friendly, and patient demeanor
    
    You may have additional context that you should use to help answer the customer's query. It will be provided to you below:
    """

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1
)

# Create the invoice information subagent using LangGraph's pre-built ReAct agent
# This agent specializes in handling customer invoice queries and billing information
invoice_information_subagent = create_react_agent(
    llm,                           # Language model for reasoning and responses
    tools=invoice_tools,           # Invoice-specific tools for database queries
    name="invoice_information_subagent",  # Unique identifier for the agent
    prompt=invoice_subagent_prompt,       # System instructions for invoice handling
    state_schema=State,            # State schema for data flow between nodes
    checkpointer=checkpointer,     # Short-term memory for conversation context
    store=in_memory_store         # Long-term memory store for persistent data
)

import uuid

# Generate a unique thread ID for this conversation session
thread_id = uuid.uuid4()

# Define the user's question about their recent invoice and employee assistance
question = "My customer id is 1. What was my most recent invoice, and who was the employee that helped me with it?"

# Set up configuration with the thread ID for maintaining conversation context
config = {"configurable": {"thread_id": thread_id}}

# Invoke the invoice information subagent with the user's question
# The agent will use its tools to search for invoice information and employee details
result = invoice_information_subagent.invoke({"messages": [HumanMessage(content=question)]}, config=config)

# Display all messages from the conversation in a formatted way
for message in result["messages"]:
    message.pretty_print()