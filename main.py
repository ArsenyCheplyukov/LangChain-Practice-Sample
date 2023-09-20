from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenAI
import time
from passwords import OPENAI_PASSWORD, LIAMA_PASSWORD

# First, let's load the language model we're going to use to control the agent.
llm = OpenAI(
    temperature=0,
    openai_api_key=OPENAI_PASSWORD,
)

# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
tools = load_tools(
    ["llm-math", "serpapi"],
    serpapi_api_key=LIAMA_PASSWORD, 
    llm=llm,
)


# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# Now let's test it out!
time.sleep(4.0)
agent.run(
    "What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?"
)
