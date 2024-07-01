# WrokFlow.py

import os
import json
import configparser
from typing import Dict, List, TypedDict, Annotated
import operator
from NodeData import NodeData
from langchain_community.llms import Ollama

from langgraph.graph import StateGraph, END, START


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


def RunWorkFlow(node_map: Dict[str, NodeData], llm):

    class PipelineState(TypedDict):
        history: Annotated[str, operator.add]
        task: Annotated[str, operator.add]

    # Define the state machine
    workflow = StateGraph(PipelineState)

    # Start node, only one start point
    start_node = find_nodes_by_type(node_map, "START")[0]
    print(f"Start root ID: {start_node.uniq_id}")

    # Step nodes
    step_nodes = find_nodes_by_type(node_map, "STEP")
    for current_node in step_nodes:
        workflow.add_node(current_node.uniq_id, lambda state: print(f"Processing step_node: {current_node.name} {current_node.uniq_id}"))


    # Condition nodes



    # edges
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


    initial_state = PipelineState(
        history="",
        task=""
    )

    app = workflow.compile()
    for state in app.stream(initial_state):
        print(state)


def run_workflow_from_file(filename: str, llm):
    node_map = load_nodes_from_json(filename)
    RunWorkFlow(node_map, llm)