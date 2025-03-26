import json

import os
from utils import generate_response
import config as config

# Construct prompt to generate the daily plan
def construct_prompt(template_path, persona):
    location = persona['location']
    personality_traits = persona.get("personality_traits", {}).get("character_traits", [])
    lifestyle = persona.get("lifestyle", "")
    daily_routine = persona.get("daily_routine", [])
    hobbies = persona.get("hobbies", [])
    
    # Construct the personality traits string
    personality_str = ', '.join(personality_traits) if personality_traits else "unique personality traits"
    
    # Construct the daily routine string
    routine_str = 'The daily routine includes: ' + ', '.join(daily_routine) if daily_routine else "You have a flexible daily routine."
    
    # Construct the hobbies string
    hobbies_str = 'The character enjoys hobbies such as ' + ', '.join(hobbies) + '.' if hobbies else "You have various hobbies."
    
    # Generate the prompt
    f = open(template_path, "r")
    prompt = f.read()
    prompt = prompt.replace(f"<personality_traits>", personality_str)
    prompt = prompt.replace(f"<location>", location)
    prompt = prompt.replace(f"<lifestyle>", lifestyle)
    prompt = prompt.replace(f"<daily_routine>", routine_str)
    prompt = prompt.replace(f"<hobbies>", hobbies_str)
    
    return prompt

# Function to generate daily plan by input persona data
def generate_daily_plan_from_persona(persona_data):
    # Generate daily plan using the prompt
    prompt_path = os.path.join(config.PROMPT_PATH, "gen_daily_plan_v2.txt")
    p = construct_prompt(prompt_path, persona_data)
    response = generate_response.generate_request(p)
    response = response.replace("```", "")
    response = response.replace("json", "")
    raw_json_object = json.loads(response)
    persona_data["daily_plan"] = raw_json_object["daily_plan"]
    return persona_data

if __name__ == "__main__":
    p_uid = "920878fc" # test case uid
    persona_file =  os.path.join(config.PERSONA_PATH, "persona_{}.json".format(p_uid))
    with open(persona_file, 'r') as json_file:
        persona_data = json.load(json_file)


    # Update persona profile with daily plan
    daily_plan_json = generate_daily_plan_from_persona(persona_data)
    persona_data["daily_plan"] = daily_plan_json["daily_plan"]
    with open(persona_file, 'w') as json_file:
        json.dump(persona_data, json_file, indent=4)
    print(f"Persona information has been updated with addition to daily plan")