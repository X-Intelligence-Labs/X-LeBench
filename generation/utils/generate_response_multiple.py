
from openai import OpenAI
import google.ai.generativelanguage as glm
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError
from http import HTTPStatus
import dashscope
import sys
import os
# os.environ['CUDA_VISIBLE_D/EVICES'] = '5'

# Add the path of the directory three levels up to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import generation.config as config
import torch
from transformers import pipeline, AutoModel, AutoTokenizer
import time

import requests
from requests.exceptions import ReadTimeout


def get_safety_settings():
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

class gen_response_mult():
    def __init__(self, tools = config.GENERATE_WAY) -> None:
        # Load model once
        self.tools = tools
        self.history = None
        if self.tools == 'gemini':
            model_name = config.GEN_MODEL
            genai_api_key = config.GEMINI_API_KEY
            genai.configure(api_key=genai_api_key, transport='rest')
            self.model = genai.GenerativeModel(model_name, safety_settings = get_safety_settings())
            self.tmpr = config.MODEL_TEMPERATURE
        elif tools == "qwen":
            dashscope.api_key = config.QWEN_API_KEY
        elif tools == "openai":
            self.client = OpenAI(
                                    # This is the default and can be omitted
                                    api_key=config.OPENAI_API_KEY,
                                )
            self.model = config.GEN_MODEL
        else:
            raise ValueError("Generat tools is not correctly defined, GENERATE_WAY must be gemini or minicpm")
        
    def gen_response(self, prompt, format_r = None, out_type = None):
        if self.tools == 'gemini':
            try: 
                response = self.model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=self.tmpr, response_mime_type=out_type, response_schema=format_r))
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
                return self.gen_response(prompt)
            except requests.exceptions.ConnectionError as e:
                print("Connection Error:", e)
                print("Retrying request...")
                return self.gen_response(prompt)
            except genai.types.generation_types.BlockedPromptException as e:
                print(f"Prompt blocked due to: {e}")
                return self.gen_response(prompt)
            except requests.exceptions.HTTPError as err:
                if response.status_code == 400:
                    print("Error 400: Bad request. The request was invalid.")
                else:
                    print(f"HTTP error occurred: {err}")
                return self.gen_response(prompt)
            except GoogleAPIError as e:
                if e.code == 429: 
                    print("ResourceExhausted: ")
                    time.sleep(10)
                    return self.gen_response(prompt)
                else:
                    print("An unexpected Google API error occurred: ", e)
                    return ""
                    # return make_request(content, other_logger)
            except Exception as e:
                print("An unexpected error occurred: ", e)
                return self.gen_response(prompt)

        elif self.tools == 'qwen':
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
                                model= config.GEN_MODEL,
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
                return self.gen_response(prompt)
        elif self.tools == 'openai':
            try: 
                response = self.client.chat.completions.create(
                    model=self.model,  
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature= config.MODEL_TEMPERATURE
                )
                print(response.choices[0].message.content)
                return response.choices[0].message.content
            except ReadTimeout as e:
                print("Timeout error occurred:", e)
                print("Retrying request...")
                return self.gen_response(prompt)
            except requests.exceptions.ConnectionError as e:
                print("Connection Error:", e)
                print("Retrying request...")
                return self.gen_response(prompt)
            except Exception as e:
                print("An unexpected error occurred: ", e)
                return self.gen_response(prompt)
    

    def multi_round_response(self, prompt):
        if self.history is None:
            self.chat = self.model.start_chat()
        try: 
            response = self.chat.send_message(prompt, generation_config=genai.types.GenerationConfig(temperature=self.tmpr))
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
            self.history = self.chat.history
            return generated_text.strip()
        except ReadTimeout as e:
            print("Timeout error occurred:", e)
            print("Retrying request...")
            return self.multi_round_response(prompt)
        except requests.exceptions.ConnectionError as e:
            print("Connection Error:", e)
            print("Retrying request...")
            return self.multi_round_response(prompt)
        except genai.types.generation_types.BlockedPromptException as e:
            print(f"Prompt blocked due to: {e}")
            return self.multi_round_response(prompt)
        except requests.exceptions.HTTPError as err:
            if response.status_code == 400:
                print("Error 400: Bad request. The request was invalid.")
            else:
                print(f"HTTP error occurred: {err}")
            return self.multi_round_response(prompt)
        except GoogleAPIError as e:
            if e.code == 429: 
                print("ResourceExhausted: ")
                time.sleep(10)
                return self.multi_round_response(prompt)
            else:
                print("An unexpected Google API error occurred: ", e)
                return ""
                # return make_request(content, other_logger)
        except Exception as e:
            print("An unexpected error occurred: ", e)
            return self.multi_round_response(prompt)
    
    def upload_files_to_gemini(self, file_list, start_idx = 0, upload_files = []):
        for idx, file in enumerate(file_list[start_idx:]):
            try: 
                myfile = genai.upload_file(file)
            except requests.exceptions.ConnectionError as e:
                print("Connection Error:", e)
                print("Retrying request...")
                return self.upload_files_to_gemini(file_list, start_idx = idx+start_idx, upload_files = upload_files)
            except requests.exceptions.HTTPError as err:
                if myfile.status_code == 400:
                    print("Error 400: Bad request. The request was invalid.")
                else:
                    print(f"HTTP error occurred: {err}")
                return self.upload_files_to_gemini(file_list, start_idx = idx+start_idx, upload_files = upload_files)
            except GoogleAPIError as e:
                if e.code == 429: 
                    print("ResourceExhausted: ")
                    time.sleep(10)
                    return self.upload_files_to_gemini(file_list, start_idx = idx+start_idx, upload_files = upload_files)
                else:
                    print("An unexpected Google API error occurred: ", e)
                    return ""
                # return make_request(content, other_logger)
            except Exception as e:
                print("An unexpected error occurred: ", e)
                return self.upload_files_to_gemini(file_list, start_idx = idx+start_idx, upload_files = upload_files)
            # print(f"{myfile=}")
            upload_files.append(myfile)
        print("My files:")
        for f in upload_files:
            print(" {}, {}".format(f.name, f.display_name))
        return upload_files

        
    def delete_files(self, files):
        if files =="all":
            for f in genai.list_files():
                print("Deleted  {}, {}".format(f.name, f.display_name))
                f.delete()
        else:
            for f in files:
                print("Deleted  {}, {}".format(f.name, f.display_name))
                f.delete()

    def list_files(self):
        cloud_files = []
        for f in genai.list_files():
            print("{}, {}".format(f.name, f.display_name))
            cloud_files.append(f)
        return cloud_files




