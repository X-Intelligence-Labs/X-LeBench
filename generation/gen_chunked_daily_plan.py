import json
import os
from utils import generate_response
import generation.config as config

# Construct prompt to segment the daily agenda into chunks
def construct_prompt(template_path, persona, ref = None):
    f = open(template_path, "r")
    prompt = f.read()
    daily_plan = persona["daily_plan"]
    daily_plan_str = "\n".join([str(item) for item in daily_plan])

    prompt = prompt.replace("<daily_plan>", f"{daily_plan_str}")
    prompt = prompt.replace("<chunk_num>", f"{config.CHUNK_NUM}")
    # prompt = prompt.replace("<occupation>", f"{occupation}")
    # prompt = prompt.replace("<hobbies>", f"{hobbies}")
    if ref is not None:
        consolidated_activities_list = ref["consolidated_activities_list"]
        prompt = prompt.replace("<activities_list>", f"{consolidated_activities_list}")
    return prompt

# Function to generate daily plan chunks
def generate_daily_plan_chunk(persona_data):
    prompt_path = os.path.join(config.PROMPT_PATH, "gen_daily_plan_chunk_v6.txt")
    p = construct_prompt(prompt_path, persona_data)

    # Generate daily plan chunks using the prompt
    response = generate_response.generate_request(p)
    response = response.replace("```", "")
    response = response.replace("json", "")
    raw_json_object = json.loads(response)
    if len(raw_json_object['daily_plan_chunk']) < config.CHUNK_NUM:
        return generate_daily_plan_chunk(persona_data)
    persona_data["daily_plan_chunk"] = raw_json_object["daily_plan_chunk"]
    return persona_data

if __name__ == "__main__":
    p_uid = "920878fc" # test case uid
    persona_file = os.path.join(config.PERSONA_PATH, "persona_{}.json".format(p_uid))
    
    with open(persona_file, 'r') as json_file:
        persona_data = json.load(json_file)

    # Update persona profile with daily plan chunks
    daily_plan_chunk_json = generate_daily_plan_chunk(persona_data)
    persona_data["daily_plan_chunk"] = daily_plan_chunk_json["daily_plan_chunk"]
    with open(persona_file, 'w') as json_file:
        json.dump(persona_data, json_file, indent=4)
    print(f"Persona information has been updated with addition to daily plan chunk")

