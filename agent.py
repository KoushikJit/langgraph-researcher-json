from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage, SystemMessage
from langgraph.graph import add_messages, StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import TavilySearchResults



from langchain_experimental.utilities import PythonREPL
import json


tavily_search = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

@tool
def json_validator(string: str):
    """Validates if the given string is a correctly formatted JSON."""
    try:
        json.loads(string)
        return "Valid JSON"
    except json.JSONDecodeError as e:
        return "Invalid JSON: " +  str(e)

llm_generic = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)
llm_json = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

research_agent = create_react_agent(
    llm_generic,
    tools=[tavily_search],
    state_modifier="You are best at researching a topic. You should do a thorough research. Your output will be used by a chart generator agent to visually display the data. Hence you should provide accurate data. Also specify the chart types like barchart, pie chart etc. that will effectively display the data. The chart generator may ask for more information, so be prepared to do further research and provide it."
)

chart_agent = create_react_agent(
    llm_json,
    tools=[json_validator],
    state_modifier="""
        You are a chart_generator agent. Your task is to create a JSON array representing charts based on data and specifications provided by the researcher agent. 

        1. **Analyze Input**: 
        - If data is incomplete, request missing details, specifying exactly what is needed. It is important to include the text ~QUESTION_TO_RESEARCHER~ in your response when asking for missing information.

        - Avoid asking for already provided information.

        2. **Generate JSON**:
        - Create an array where each object represents a chart. Here is a sample output:
            ```json
            {
                chartsArray: [
                    {
                        type: "bar",
                        title: "Top EDM Songs by Streaming Numbers",
                        xAxisLabel: "Streaming Numbers (in billions)",
                        yAxisLabel: "Songs",
                        descriptionOfLabel: "song",
                        chartData: [
                            { label: "Levels", value: 380000000 },
                            { label: "Titanium", value: 1500000000 },
                            { label: "Wake Me Up", value: 1500000000 },
                            { label: "Animals", value: 1000000000 },
                            { label: "This Is What It Feels Like", value: 300000000 }
                        ],
                        chartFootNote: "<Go through the research agent's output and include all research findings that pertain to this chart here>"
                    },
                    {
                        type: "pie",
                        title: "Distribution of Top EDM Songs by Artist",
                        descriptionOfLabel: "artist",
                        chartData: [
                            { label: "Avicii", value: 40 },
                            { label: "David Guetta", value: 30 },
                            { label: "Martin Garrix", value: 30}
                        ],
                        chartFootNote: "<Go through the research agent's output and include all research findings that pertain to this chart here>"
                    }
                ],
                otherResearchFindings: "<Go through the research agent's output and include any research findings that were not already included in the chart foot notes>"
            }
            ```

        3. **Validate Output**:
        - Use the `json_validator` tool to ensure the JSON is valid.
        - Regenerate JSON until valid.

        **Output only the JSON in the specified format, without any additional text.**
    """
)  

class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

graph_builder = StateGraph(GraphState)
graph_builder.add_node("research_node", research_agent)
graph_builder.add_node("chart_node", chart_agent)

graph_builder.add_edge(START, "research_node")
graph_builder.add_edge("research_node", "chart_node")
# todo
def chart_to_research_condition(state: GraphState) -> str:
    chart_content = state["messages"][-1].content
    if "QUESTION_TO_RESEARCHER" in chart_content:
        return "research_more"
    else:
        return "path_end"
    
graph_builder.add_conditional_edges(
    "chart_node", 
    chart_to_research_condition, 
    {"research_more": "research_node", "path_end": END}
)
# end todo
graph = graph_builder.compile()
