#!/bin/bash

available_scripts=(
  "agent1" "agent2" "agent3" "agent4" "agent5"
  "workflow1" "workflow2"
)

cur_script="${available_scripts[6]}"

python -m "my_demos.langgraph.get_started.${cur_script}"