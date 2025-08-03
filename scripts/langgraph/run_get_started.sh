#!/bin/bash

available_scripts=(
  "agent1" "agent2" "agent3" "agent4" "agent5"
  "workflow1" "workflow2" "workflow3"
)

cur_script="${available_scripts[7]}"

python -m "my_demos.langgraph.get_started.${cur_script}"