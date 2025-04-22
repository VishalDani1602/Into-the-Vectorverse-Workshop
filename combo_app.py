# =============================================================
# STEP 1: Import Dependencies and Load Environment
# =============================================================

import os
import streamlit as st
from dotenv import load_dotenv
from typing import Type
from pydantic import BaseModel
from crewai import Agent, Crew, Task, Process
from crewai.tools import BaseTool
from opik.integrations.crewai import track_crewai
from opik.integrations.openai import track_openai
import opik
from openai import OpenAI

# Load environment variables from .env file (e.g. API keys)
load_dotenv()

# Configure logging and tracking via Opik and OpenAI
opik.configure(use_local=False, api_key=os.getenv("OPIK_API_KEY"))
openai_client = track_openai(OpenAI())
track_crewai(project_name="Interactive-Workshop")


# =============================================================
# STEP 2: Define a Custom Weaviate Tool
# =============================================================

# Define input schema for the tool using Pydantic
class WeaviateSearchInput(BaseModel):
    topic: str 

# Custom tool that queries Weaviate using semantic search
class CustomWeaviateTool(BaseTool):
    name: str = "Weaviate Book Search Tool"
    description: str = "Searches Weaviate for relevant books related to a given topic."
    args_schema: Type[BaseModel] = WeaviateSearchInput

    @opik.track  # Log usage through Opik
    def _run(self, topic: str) -> str:
        import weaviate
        from weaviate.classes.init import Auth

        try:
            # Connect to your Weaviate instance
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=os.getenv("WEAVIATE_CLUSTER_URL"),
                auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
                headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
            )

            # Search for similar books using the "Book" collection
            collection = client.collections.get(name="Book")
            results = collection.query.near_text(query=topic, limit=3)

            if not results.objects:
                return "No books found in Weaviate."

            # Format and return top book titles
            return "\n".join(f"📘 {obj.properties['title']}" for obj in results.objects)

        except Exception as e:
            return f"❌ Error querying Weaviate: {e}"


# =============================================================
# STEP 3: Define Agents and Tasks
# =============================================================

class WorkshopAgentsAndTasks:
    def __init__(self, user_query):
        self.user_query = user_query
        self.weaviate_tool = CustomWeaviateTool()  # Instantiates the tool
        self.data_analyst_agent = self.create_data_analyst_agent()
        self.resource_recommender_agent = self.create_resource_recommender_agent()

    # Agent that performs trend analysis
    def create_data_analyst_agent(self) -> Agent:
        return Agent(
            role="Data Analyst",
            goal=f"Generate insights and trends about '{self.user_query}'.",
            backstory="An experienced data professional skilled in extracting meaning from complex datasets and market data.",
            verbose=True,
            model="o3-mini",
        )

    # Agent that uses the Weaviate tool to find resources
    def create_resource_recommender_agent(self) -> Agent:
        return Agent(
            role="Resource Recommender",
            goal="Provide relevant books and resources based on semantic search.",
            backstory="An expert in research and knowledge systems, skilled at surfacing valuable content using tools like Weaviate.",
            tools=[self.weaviate_tool],
            verbose=True,
            model="o3-mini",
        )

    # Task assigned to the Data Analyst agent
    def create_data_analysis_task(self) -> Task:
        return Task(
            name="Trend Analysis",
            description=f"Analyze key patterns and trends related to '{self.user_query}' using your knowledge and reasoning abilities.",
            expected_output="A structured summary with 2-3 key insights or trends.",
            agent=self.data_analyst_agent,
        )

    # Task assigned to the Resource Recommender agent
    def create_resource_recommendation_task(self) -> Task:
        return Task(
            name="Book Finder Task",
            description=(
                f"Use only the Weaviate tool to find high-quality books or articles about '{self.user_query}'. "
                f"You must base your answer strictly on the output returned by the tool and do not use external knowledge or suggestions. "
                f"If the tool does not return any relevant results, say so."
            ),
            expected_output=(
                "A list of up to 3 recommended books or resources returned from Weaviate, each with a short explanation. "
                "Do not fabricate or include items not returned by the tool."
            ),
            agent=self.resource_recommender_agent,
        )



    # Creates the full crew that runs the tasks
    def create_workshop_crew(self) -> Crew:
        return Crew(
            agents=[self.data_analyst_agent, self.resource_recommender_agent],
            tasks=[self.create_data_analysis_task(), self.create_resource_recommendation_task()],
            process=Process.sequential,  # Run tasks one after another
            verbose=True,
        )


# =============================================================
# STEP 4: Streamlit UI
# =============================================================

# Set page title and layout
st.set_page_config(page_title="CrewAI + Weaviate", layout="centered")
st.title("🧠 CrewAI + Weaviate + Comet")

# Input for user to type a query
user_query = st.text_input("Enter a topic or query you'd like analyzed and researched:")

# Run CrewAI workflow when button is clicked
if st.button("Run Analysis") and user_query:
    st.info("Running CrewAI Workflow... Please wait ⏳")

    workshop = WorkshopAgentsAndTasks(user_query)
    crew = workshop.create_workshop_crew()

    try:
        result = crew.kickoff()  # Executes all tasks in order
        st.success("✅ Workflow completed successfully!")

        # Display output of Data Analyst agent (task 1)
        st.markdown("### 📊 Data Analyst Output")
        st.markdown(crew.tasks[0].output)

        # Display output of Resource Recommender agent (final result)
        st.markdown("### 📚 Resource Recommender Output")
        st.markdown(result)

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
