import os
from dotenv import load_dotenv
from langsmith import utils
import llm.tools

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
def get_albums_by_artist(artist: str):
    """
    Get albums by an artist from the music database.
    
    Args:
        artist (str): The name of the artist to search for albums.
    
    Returns:
        str: Database query results containing album titles and artist names.
    """
    return db.run(
        f"""
        SELECT Album.Title, Artist.Name
        FROM Album
        JOIN Artist ON Album.ArtistId = Artist.ArtistId
        WHERE Artist.Name LIKE '%{artist}%';
        """,
        include_columns=True
    )

@tool
def get_tracks_by_artist(artist: str):
    """
    Get songs/tracks by an artist (or similar artists) from the music database.
    
    Args:
        artist (str): The name of the artist to search for tracks.
    
    Returns:
        str: Database query results containing song names and artist names.
    """
    return db.run(
        f"""
        SELECT Track.Name as SongName, Artist.Name as ArtistName 
        FROM Album 
        LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId 
        LEFT JOIN Track ON Track.AlbumId = Album.AlbumId 
        WHERE Artist.Name LIKE '%{artist}%';
        """,
        include_columns=True
    )

@tool
def get_songs_by_genre(genre: str):
    """
    Fetch songs from the database that match a specific genre.
    
    This function first looks up the genre ID(s) for the given genre name,
    then retrieves songs that belong to those genre(s), limiting results
    to 8 songs grouped by artist.
    
    Args:
        genre (str): The genre of the songs to fetch.
    
    Returns:
        list[dict] or str: A list of songs with artist information that match 
                          the specified genre, or an error message if no songs found.
    """
    # First, get the genre ID(s) for the specified genre
    genre_id_query = f"SELECT GenreId FROM Genre WHERE Name LIKE '%{genre}%'"
    genre_ids = db.run(genre_id_query)
    
    # Check if any genres were found
    if not genre_ids:
        return f"No songs found for the genre: {genre}"
    
    # Parse the genre IDs and format them for the SQL query
    genre_ids = ast.literal_eval(genre_ids)
    genre_id_list = ", ".join(str(gid[0]) for gid in genre_ids)

    # Query for songs in the specified genre(s)
    songs_query = f"""
        SELECT Track.Name as SongName, Artist.Name as ArtistName
        FROM Track
        LEFT JOIN Album ON Track.AlbumId = Album.AlbumId
        LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId
        WHERE Track.GenreId IN ({genre_id_list})
        GROUP BY Artist.Name
        LIMIT 8;
    """
    songs = db.run(songs_query, include_columns=True)
    
    # Check if any songs were found
    if not songs:
        return f"No songs found for the genre: {genre}"
    
    # Format the results into a structured list of dictionaries
    formatted_songs = ast.literal_eval(songs)
    return [
        {"Song": song["SongName"], "Artist": song["ArtistName"]}
        for song in formatted_songs
    ]

@tool
def check_for_songs(song_title):
    """
    Check if a song exists in the database by its name.
    
    Args:
        song_title (str): The title of the song to search for.
    
    Returns:
        str: Database query results containing all track information 
             for songs matching the given title.
    """
    return db.run(
        f"""
        SELECT * FROM Track WHERE Name LIKE '%{song_title}%';
        """,
        include_columns=True
    )

music_tools = [
    get_albums_by_artist,
    get_tracks_by_artist,
    get_songs_by_genre, 
    check_for_songs
]
from langchain_openai import ChatOpenAI
llm = ChatOpenAI()
llm_with_music_tools = llm.bind_tools(music_tools)

from langgraph.prebuilt import ToolNode

# Create a tool node that executes the music-related tools
# ToolNode is a pre-built LangGraph component that handles tool execution
music_tool_node = ToolNode(music_tools)

from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

def generate_music_assistant_prompt(memory: str = "None") -> str:
    """
    Generate a system prompt for the music assistant agent.
    
    Args:
        memory (str): User preferences and context from long-term memory store
        
    Returns:
        str: Formatted system prompt for the music assistant
    """
    return f"""
    You are a member of the assistant team, your role specifically is to focused on helping customers discover and learn about music in our digital catalog. 
    If you are unable to find playlists, songs, or albums associated with an artist, it is okay. 
    Just inform the customer that the catalog does not have any playlists, songs, or albums associated with that artist.
    You also have context on any saved user preferences, helping you to tailor your response. 
    
    CORE RESPONSIBILITIES:
    - Search and provide accurate information about songs, albums, artists, and playlists
    - Offer relevant recommendations based on customer interests
    - Handle music-related queries with attention to detail
    - Help customers discover new music they might enjoy
    - You are routed only when there are questions related to music catalog; ignore other questions. 
    
    SEARCH GUIDELINES:
    1. Always perform thorough searches before concluding something is unavailable
    2. If exact matches aren't found, try:
       - Checking for alternative spellings
       - Looking for similar artist names
       - Searching by partial matches
       - Checking different versions/remixes
    3. When providing song lists:
       - Include the artist name with each song
       - Mention the album when relevant
       - Note if it's part of any playlists
       - Indicate if there are multiple versions
    
    Additional context is provided below: 

    Prior saved user preferences: {memory}
    
    Message history is also attached.  
    """
def music_assistant(state: State, config: RunnableConfig):
    """
    Music assistant node that handles music catalog queries and recommendations.
    
    This node processes customer requests related to music discovery, album searches,
    artist information, and personalized recommendations based on stored preferences.
    
    Args:
        state (State): Current state containing customer_id, messages, loaded_memory, etc.
        config (RunnableConfig): Configuration for the runnable execution
        
    Returns:
        dict: Updated state with the assistant's response message
    """
    # Retrieve long-term memory preferences if available
    memory = "None" 
    if "loaded_memory" in state: 
        memory = state["loaded_memory"]

    # Generate instructions for the music assistant agent
    music_assistant_prompt = generate_music_assistant_prompt(memory)

    # Invoke the language model with tools and system prompt
    # The model can decide whether to use tools or respond directly
    response = llm_with_music_tools.invoke([SystemMessage(music_assistant_prompt)] + state["messages"])
    
    # Return updated state with the assistant's response
    return {"messages": [response]}

def should_continue(state: State, config: RunnableConfig):
    """
    Conditional edge function that determines the next step in the ReAct agent workflow.
    
    This function examines the last message in the conversation to decide whether the agent
    should continue with tool execution or end the conversation.
    
    Args:
        state (State): Current state containing messages and other workflow data
        config (RunnableConfig): Configuration for the runnable execution
        
    Returns:
        str: Either "continue" to execute tools or "end" to finish the workflow
    """
    # Get all messages from the current state
    messages = state["messages"]
    
    # Examine the most recent message to check for tool calls
    last_message = messages[-1]
    
    # If the last message doesn't contain any tool calls, the agent is done
    if not last_message.tool_calls:
        return "end"
    # If there are tool calls present, continue to execute them
    else:
        return "continue"
    
from langgraph.graph import StateGraph, START, END
from langsmith import utils
from langsmith import show_graph
# Create a new StateGraph instance for the music workflow
music_workflow = StateGraph(State)

# Add nodes to the graph
# music_assistant: The reasoning node that decides which tools to invoke or responds directly
music_workflow.add_node("music_assistant", music_assistant)
# music_tool_node: The execution node that handles all music-related tool calls
music_workflow.add_node("music_tool_node", music_tool_node)

# Add edges to define the flow of the graph
# Set the entry point - all queries start with the music assistant
music_workflow.add_edge(START, "music_assistant")

# Add conditional edge from music_assistant based on whether tools need to be called
music_workflow.add_conditional_edges(
    "music_assistant",
    # Conditional function that determines the next step
    should_continue,
    {
        # If tools need to be executed, route to tool node
        "continue": "music_tool_node",
        # If no tools needed, end the workflow
        "end": END,
    },
)

# After tool execution, always return to the music assistant for further processing
music_workflow.add_edge("music_tool_node", "music_assistant")

# Compile the graph with checkpointer for short-term memory and store for long-term memory
music_catalog_subagent = music_workflow.compile(
    name="music_catalog_subagent", 
    checkpointer=checkpointer, 
    store=in_memory_store
)

# Display the compiled graph structure
show_graph(music_catalog_subagent)

import uuid

# Generate a unique thread ID for this conversation session
thread_id = uuid.uuid4()

# Define the user's question about music recommendations
question = "I like the Rolling Stones. What songs do you recommend by them or by other artists that I might like?"

# Set up configuration with the thread ID for maintaining conversation context
config = {"configurable": {"thread_id": thread_id}}

# Invoke the music catalog subagent with the user's question
# The agent will use its tools to search for Rolling Stones music and provide recommendations
result = music_catalog_subagent.invoke({"messages": [HumanMessage(content=question)]}, config=config)

# Display all messages from the conversation in a formatted way
for message in result["messages"]:
   message.pretty_print()