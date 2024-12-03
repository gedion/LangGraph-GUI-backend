import os
import json
import re
from typing import Dict, List, TypedDict, Annotated, Callable, Literal
from langchain_community.chat_models import ChatOpenAI, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END, START
from dotenv import load_dotenv
from NodeData import NodeData
from IPython.display import Image, display
import inspect

# Load variables from .env file
load_dotenv()

# Tool registry to hold information about tools
tool_registry: Dict[str, Callable] = {}
tool_info_registry: Dict[str, str] = {}

import base64

def get_image_tag_from_binary(png_binary: bytes) -> str:
    """
    Converts PNG binary data into a Base64-encoded image tag.
    """
    # Encode binary data as Base64
    base64_data = base64.b64encode(png_binary).decode("utf-8")
    
    # Create the HTML <img> tag
    img_tag = f'<img src="data:image/png;base64,{base64_data}" alt="Graph Visualization">'
    return img_tag

# Decorator to register tools
def tool(func: Callable) -> Callable:
    signature = inspect.signature(func)
    docstring = func.__doc__ or ""
    tool_info = f"{func.__name__}{signature} - {docstring}"
    tool_registry[func.__name__] = func
    tool_info_registry[func.__name__] = tool_info
    return func

def load_nodes_from_json(filename: str) -> Dict[str, NodeData]:
    with open(filename, 'r') as file:
        data = json.load(file)
        node_map = {}
        for node_data in data["nodes"]:
            node = NodeData.from_dict(node_data)
            node_map[node.uniq_id] = node
        return node_map

def find_nodes_by_type(node_map: Dict[str, NodeData], node_type: str) -> List[NodeData]:
    return [node for node in node_map.values() if node.type == node_type]

# Define State with messages
class State(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]  # Append new messages to the list
    condition: Annotated[bool, lambda x, y: y]

def conditional_edge(state: State) -> Literal["True", "False"]:
    if state["condition"] in ["True", "true", True]:
        return "True"
    else:
        return "False"
# Function to handle node execution
def execute_step(name: str, state: State, prompt_template: str, llm) -> State:
    print(f"{name} is working...")
    messages = state["messages"]

    # Generate prompt from messages
    prompt = PromptTemplate.from_template(prompt_template)
    llm_chain = prompt | llm | StrOutputParser()
    inputs = {"messages": messages}
    print('inputs')
    generation = llm_chain.invoke(inputs)
    # Add new message to state
    new_message = json.loads(generation)
    print('new_message ', new_message)
    state["messages"].append(("system", new_message))
    return state

def create_step_node_executor(name: str, template: str, llm):
    """
    Creates a step node executor function.
    """
    def executor(state: State):
        return execute_step(name, state, template, llm)
    return executor

def condition_switch(name: str, state: State, prompt_template: str, llm) -> State:
    print(f"{name} is working...")

    # Extract and format messages for the prompt
    messages = state["messages"]
    
    # Generate the prompt
    prompt = PromptTemplate.from_template(prompt_template)
    llm_chain = prompt | llm | StrOutputParser()
    inputs = {"messages": messages}
    generation = llm_chain.invoke(inputs)

    # Parse the LLM output
    data = json.loads(generation)
    condition = data.get("switch", False)

    # Update state with the condition result and new system message
    state["messages"].append(("system", f"Condition is {condition}"))
    state["condition"] = condition

    return state


# Workflow runner
def RunWorkFlow(node_map: Dict[str, NodeData], llm):
    # Define the state machine
    workflow = StateGraph(State)

    # Start node, only one start point
    start_node = find_nodes_by_type(node_map, "START")[0]
    print('start_node ', start_node)
    print(f"Start root ID: {start_node.uniq_id}")

    # Step nodes
    step_nodes = find_nodes_by_type(node_map, "STEP")
    for current_node in step_nodes:
        if current_node.tool:
            tool_info = tool_info_registry[current_node.tool]
            prompt_template = f"""
            messages: {{messages}}
            {current_node.description}
            Available tool: {tool_info}
            Based on Available tool, arguments in the json format:
            "function": "<func_name>", "args": [<arg1>, <arg2>, ...]

            next stage directly parse then run <func_name>(<arg1>,<arg2>, ...) make sure syntax is right json and align function siganture
            """
            workflow.add_node(
                current_node.uniq_id,
                create_step_node_executor(
                    name=current_node.name,
                    template=prompt_template,
                    llm=llm
                )
            )
        else:
            prompt_template = f"""
            messages: {{messages}}
            {current_node.description}
            """
            workflow.add_node(
                current_node.uniq_id,
                create_step_node_executor(
                    name=current_node.name,
                    template=prompt_template,
                    llm=llm
                )
            )
    # Edges
    # Find all next nodes from start_node
    next_node_ids = start_node.nexts
    next_nodes = [node_map[next_id] for next_id in next_node_ids]
    
    for next_node in next_nodes:
        print(f"Next node ID: {next_node.uniq_id}, Type: {next_node.type}")
        workflow.add_edge(START, next_node.uniq_id)   

    # Find all next nodes from step_nodes
    for node in step_nodes:
        next_nodes = [node_map[next_id] for next_id in node.nexts]
        
        for next_node in next_nodes:
            print(f"{node.name} {node.uniq_id}'s next node: {next_node.name} {next_node.uniq_id}, Type: {next_node.type}")
            workflow.add_edge(node.uniq_id, next_node.uniq_id)

    # Find all condition nodes
    condition_nodes = find_nodes_by_type(node_map, "CONDITION")
    for condition in condition_nodes:
        condition_template = f"""{condition.description}
        messages: {{messages}}, decide the condition result in the json format:
        "switch": True/False
        """
        workflow.add_node(
            condition.uniq_id, 
            lambda state, template=condition_template, llm=llm, name=condition.name: condition_switch(name, state, template, llm)
        )

        print(f"{condition.name} {condition.uniq_id}'s condition")
        print(f"true will go {condition.true_next}")
        print(f"false will go {condition.false_next}")
        workflow.add_conditional_edges(
            condition.uniq_id,
            conditional_edge,
            {
                "True": condition.true_next if condition.true_next else END,
                "False": condition.false_next if condition.false_next else END
            }
        )
        
    initial_state = State(messages=[("user", "hello")])
    app = workflow.compile()
    png_binary = get_image_tag_from_binary(app.get_graph().draw_mermaid_png())
    print(png_binary)
    for state in app.stream(initial_state):
        print(state)

def run_workflow_as_server(params):
    print('params ', params['file'])
    node_map = load_nodes_from_json(params['file'])
    print('node_map ', node_map)
    # Register tool functions dynamically
    for tool in find_nodes_by_type(node_map, "TOOL"):
        tool_code = f"{tool.description}"
        exec(tool_code, globals())
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0
        )

    RunWorkFlow(node_map, llm)
