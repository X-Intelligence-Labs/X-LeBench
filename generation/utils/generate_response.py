from openai import OpenAI
import google.ai.generativelanguage as glm
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError
import re
from http import HTTPStatus
import dashscope
import sys
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# Add the path of the directory three levels up to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import generation.config as config
import torch
from transformers import pipeline, AutoModel, AutoTokenizer
import time

import requests
from requests.exceptions import ReadTimeout


def get_safety_settings():
    """
    Safety settings for gemini.
    """
    safety_settings = [
        {
            "category": "HARM_CATEGORY_DANGEROUS",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]
    return safety_settings


def generate_request(prompt, tools=config.GENERATE_WAY):
    """
    Generates a textual response for a given prompt using either GEMINI or QWEN.
    
    Args:
    - prompt (str): The text prompt to generate a response for.
    - tools (str): A string indicating whether to use the GEMINI API (`gemini`), OPENAI (`openai`) or QWEN (`qwen`).
    
    Returns:
    - str: The generated text.
    """
    if tools == 'gemini':
        genai_api_key = config.GEMINI_API_KEY
        genai.configure(api_key=genai_api_key, transport='rest')
        try: 
            model = genai.GenerativeModel(config.GEN_MODEL, safety_settings = get_safety_settings())
            response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature= config.MODEL_TEMPERATURE))
            try:
                # Check if 'candidates' list is not empty
                if response.candidates:
                    # Access the first candidate's content if available
                    if response.candidates[0].content.parts:
                        generated_text = response.candidates[0].content.parts[0].text
                    else:
                        print("No generated text found in the candidate.")
                else:
                    print("No candidates found in the response.")
                    generated_text=''
            except (AttributeError, IndexError) as e:
                print("Error:", e)
                generated_text=''
            print(generated_text)
            return generated_text.strip()
        except ReadTimeout as e:
            print("Timeout error occurred:", e)
            print("Retrying request...")
            return generate_request(prompt)
        except requests.exceptions.ConnectionError as e:
            print("Connection Error:", e)
            print("Retrying request...")
            return generate_request(prompt)
        except genai.types.generation_types.BlockedPromptException as e:
            print(f"Prompt blocked due to: {e}")
            return generate_request(prompt)
        except requests.exceptions.HTTPError as err:
            if response.status_code == 400:
                print("Error 400: Bad request. The request was invalid.")
            else:
                print(f"HTTP error occurred: {err}")
            return generate_request(prompt)
        except GoogleAPIError as e:
            if e.code == 429: 
                print("ResourceExhausted: ")
                time.sleep(10)
                return generate_request(prompt)
            else:
                print("An unexpected Google API error occurred: ", e)
                return ""
                # return make_request(content, other_logger)
        except Exception as e:
            print("An unexpected error occurred: ", e)
            return generate_request(prompt)
        
    elif tools == 'openai':
        client = OpenAI(
                    api_key=config.OPENAI_API_KEY,
                )
        
        model = config.GEN_MODEL
        response = client.chat.completions.create(
            model=model,  
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=config.MODEL_TEMPERATURE
        )
        print(response.choices[0].message.content)
        return response.choices[0].message.content
    elif tools == "qwen":
        dashscope.api_key = config.QWEN_API_KEY
        messages = [{'role': 'user', 'content': prompt}]
        try :
            response = dashscope.Generation.call(model = config.GEN_MODEL,messages=messages,result_format='message', temperature = config.MODEL_TEMPERATURE)
            if response.output and response.output.choices:
                print(response.output.choices[0]['message']['content'])
                return response.output.choices[0]['message']['content']
            else:
                print("API response is invalid. Try stream now...")
                # return generate_request(prompt)
                responses = dashscope.Generation.call(
                            model=config.GEN_MODEL,
                            messages=messages,
                            result_format='message',
                            stream=True,
                            incremental_output=True,
                            temperature = config.MODEL_TEMPERATURE
                            )
                full_content = ""
                for response in responses:
                    if response.status_code == HTTPStatus.OK:
                        print(response)
                        full_content += response.output.choices[0].message.content
                    else:
                        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                            response.request_id, response.status_code,
                            response.code, response.message
                        ))
                print(f"Full content:{full_content}")
                return full_content
                
        except ReadTimeout as e:
            print("Timeout error occurred:", e)
            print("Retrying request...")
            return generate_request(prompt)
    else:
        raise ValueError("Generat tools is not correctly defined, GENERATE_WAY must be gemini, openai or qwen")
        

