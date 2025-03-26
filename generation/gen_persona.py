import json
import select
import uuid
import re
import random
from utils import generate_response
import os
import config as config

# Construct prompt to generate the persona profile
def construct_prompt(template_path, **kwargs):
    for key, value in kwargs.items():
        if key == "mbti":
            mbti_type = value
        if key == 'location':
            location = value
    f = open(template_path, "r")
    prompt = f.read()
    prompt = prompt.replace(f"<mbti_type>", mbti_type)
    prompt = prompt.replace(f"<location>", location)
    f.close()
    return prompt

# Function to generate persona by input mbti type and location
def generate_persona_from_prompt(prompt_path, mbti, loc):
    p = construct_prompt(template_path=prompt_path, mbti = mbti,location=loc)
    # Generate raw persona data using the prompt
    response = generate_response.generate_request(p)
    response = response.replace("```", "")
    response = response.replace("json", "")
    raw_json_object = json.loads(response)
    raw_json_object["persona_id"] = str(uuid.uuid4())[:8]
    raw_json_object["gen_way"] = config.GENERATE_WAY

    # Save the initial persona information to a JSON file
    persona_json = "persona_" + raw_json_object["persona_id"] + '.json'
    save_path = os.path.join(config.PERSONA_PATH, persona_json)
    with open(save_path, 'w') as json_file:
        json.dump(raw_json_object, json_file, indent=4)
    print(f"Persona information has been saved to {save_path}")
    return raw_json_object



if __name__ == "__main__":
    # Load the ego4d location JSON file
    with open('generation/ego4d_info/univ_loc_map.json', 'r') as file:
        locations = json.load(file)
    # # selected_location = sample_loc_from_ego4d_prob(location_probabilities['loc_portion'])
    for l in locations["uni_to_loc"].values():
        selected_location = l
        prompt_path = os.path.join(config.PROMPT_PATH, "gen_persona_v2.txt")
        gen_persona = generate_persona_from_prompt(prompt_path=prompt_path, mbti="ENFP", loc=selected_location)

        persona_ids = os.path.join(config.PERSONA_PATH, "persona_ids.json")
        with open(persona_ids, 'r') as file:
            p_uids = json.load(file)

        p_uids.append(gen_persona['persona_id'])
        with open(persona_ids, 'w') as file:
            json.dump(p_uids, file, indent=4)