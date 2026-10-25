import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

# Load environment variables from .env
load_dotenv()

# Step 0: User Input & Model Setup
city = input("Which city? ")

# Configure GPT-4o mini with your OpenAI API key
llm = LLM(
    model="openai/gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Step 1: Make the Agent (Role, Goal, Backstory, Verbose)
planner = Agent(
    role="Local City Guide & Travel Curator",
    goal="Provide the top 3 must-visit places for any given city with concise, practical reasons.",
    backstory=(
        "You are an experienced local travel concierge who specializes in minimalist, "
        "high-value travel recommendations. You skip the fluff and give travelers "
        "the exact highlight and the best practical tip for visiting."
    ),
    llm=llm,
    verbose=True
)

# Step 2: Make the Task (Description with {city}, Expected Output, Agent)
task = Task(
    description=(
        "Suggest exactly 3 must-visit places in {city}. For each place, provide its name "
        "and a short one-line tip or highlight explaining why to visit."
    ),
    expected_output=(
        "A clean numbered list of exactly 3 places in the following format:\n"
        "1. [Place Name] - [Short highlight/practical tip]\n"
        "2. [Place Name] - [Short highlight/practical tip]\n"
        "3. [Place Name] - [Short highlight/practical tip]"
    ),
    agent=planner
)

# Step 3: Run it in the Crew
crew = Crew(
    agents=[planner],
    tasks=[task],
    verbose=True
)

result = crew.kickoff(inputs={"city": city})
print("\n" + str(result))