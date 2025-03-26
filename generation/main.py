import json
import uuid
from sentence_transformers import SentenceTransformer
import chromadb
import random
from datetime import datetime, timedelta
import time
from utils import generate_response, generate_response_multiple
import os
import generation.config as config
import gen_persona, gen_daily_plan, gen_chunked_daily_plan, retrieve_video
import copy

def get_persona_profile(select_loc, select_mbti):
    print("-----------------------------")
    prompt_path = os.path.join(config.PROMPT_PATH, "gen_persona_v2.txt")
    persona = gen_persona.generate_persona_from_prompt(prompt_path=prompt_path, mbti=select_mbti, loc=select_loc)
    persona_ids = os.path.join(config.PERSONA_PATH, config.SAVE_PERSONA_LIST)
    with open(persona_ids, 'r') as file:
        p_uids = json.load(file)
    p_uids.append(persona['persona_id'])
    with open(persona_ids, 'w') as file:
        json.dump(p_uids, file, indent=4)
    return persona

def get_persona_daily_plan(persona_profile):
    p_uid = persona_profile['persona_id']
    persona_path = os.path.join(config.PERSONA_PATH, "persona_{}.json".format(p_uid))
    updated_persona = gen_daily_plan.generate_daily_plan_from_persona(persona_profile)
    with open(persona_path, 'w') as json_file:
        json.dump(updated_persona, json_file, indent=4)
    print(f"Persona information has been updated with addition to daily plan")
    return updated_persona

def get_chunked_daily_plan(persona_profile):
    p_uid = persona_profile['persona_id']
    persona_path = os.path.join(config.PERSONA_PATH, "persona_{}.json".format(p_uid))
    updated_persona = gen_chunked_daily_plan.generate_daily_plan_chunk(persona_profile)
    with open(persona_path, 'w') as json_file:
        json.dump(updated_persona, json_file, indent=4)
    print(f"Persona information has been updated with addition to daily plan chunk")
    return updated_persona

class life_logging_simulation_pip(object):
    def __init__(self):
        # Initialize embedding & generation model
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.gen_tool = generate_response_multiple.gen_response_mult()

        # Initialize prompt path
        self.chunk_info_prompt_path = os.path.join(config.PROMPT_PATH, "gen_chunk_ref_info.txt")
        self.reflect_prompt_path = os.path.join(config.PROMPT_PATH, "reflection_v2.txt")
        
        # Load Ego4d video info
        ref_scenario_file = os.path.join(config.EGO4DINFO_PATH, "ref_scenarios_list.json")
        with open(ref_scenario_file, 'r') as json_file:
            self.ref_sce = json.load(json_file)
        
        sum_path =os.path.join(config.EGO4DINFO_PATH, "video_info.json")
        with open(sum_path, 'r') as f:
            self.sum_data = json.load(f)

        self.map_uid = {}
        for idx, item in enumerate(self.sum_data):
            self.map_uid[item['video_uid']] = idx

        # Create/load chroma database
        self.client = chromadb.PersistentClient("/mnt/nas/wenqizhou/wenqizhou/ego4d_info/chromaDB")
        self.collection = self.client.get_collection(name='video_info_all')
        print("Database load successfully!")
    
    def life_log_simulation(self, retrieve_thr, retrieve_chunk_num, persona_profile):
        p_uid = persona_profile['persona_id']
        temp_doc = copy.deepcopy(persona_profile["daily_plan_chunk"])

        loc = persona_profile['location']
        selected_chunks = retrieve_video.evenly_sample_chunks(temp_doc, retrieve_chunk_num)
        memory_meta = {}
        memory_meta['persona_id'] = persona_profile['persona_id']
        memory_meta['memory_id'] = str(uuid.uuid4())[:8]
        memory_meta['gen_way'] = config.GENERATE_WAY
        memory_meta['priority_location'] = persona_profile['location']

        memory_path = os.path.join(config.MEMORY_PATH, "persona_{}_{}_{}.json".format(p_uid, memory_meta['memory_id'], retrieve_chunk_num))

        memory = []
        memory.append(memory_meta)

        renew_plan = []
        selected_uid = ["empty"]
        for idx, _ in enumerate(selected_chunks):
            single_plan_chunk = copy.deepcopy(selected_chunks[idx])
            chunk_ref_info = retrieve_video.generate_ref_info(self.chunk_info_prompt_path, single_plan_chunk, self.ref_sce, self.gen_tool)
            chunk_ref_info['time_period'] = retrieve_video.classify_time(single_plan_chunk['start_time'])
            chunk_ref_info['global_location'] = loc
            filtered_uid = retrieve_video.filter_uids(chunk_ref_info, selected_uid, retrieve_thr, self.collection, self.sum_data, self.map_uid)

            # Select 1 out of the top-2 results
            query = single_plan_chunk['plan_chunk']
            if len(filtered_uid)>2:
                top_k = 2
                select_idx = random.sample([0,1], 1)
            else:
                top_k = 1
                select_idx = [0]
            query_embedding = self.model.encode(query).tolist()
            results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where={
                        "video_uid": {"$in": filtered_uid},
                    }
                    )
            select_uid = results['metadatas'][0][select_idx[0]]['video_uid']
            selected_uid.append(select_uid)
            select_result = self.collection.get(
                where={"video_uid": select_uid}
            )

            for doc, metadata in zip(select_result['documents'], select_result['metadatas']):
                print(f"Consolidated Summary: {doc}, Metadata: {metadata}")
            metadata['retrieval_distance'] = results['distances'][0][select_idx[0]]
            metadata['retrieval_plan_chunk'] = query
            metadata['retrieval_time'] = single_plan_chunk['start_time']
            doc = doc.replace("C", "the character")

            # Update memory
            selected_chunks[idx]['record_content'] = doc
            selected_chunks[idx]['record_duration'] = str(metadata['video_duration']) + " seconds"
            start_time_obj = datetime.strptime(single_plan_chunk['start_time'], "%H:%M")
            end_time_obj = start_time_obj + timedelta(seconds=metadata['video_duration'])
            end_time = end_time_obj.strftime("%H:%M")
            selected_chunks[idx]['record_end_time'] = end_time
            metadata['retrieval_end_time'] = end_time

            memory.append(metadata)

            # Reflection
            if idx >= len(selected_chunks) - 1:
                break

            renew_daily_plan_chunk = retrieve_video.reflection(self.reflect_prompt_path, persona_profile, selected_chunks, idx, self.gen_tool)
            

            # Update unrecorded plan chunks
            for unrecord_id, item in enumerate(selected_chunks[idx+1:]):
                item['start_time'] = renew_daily_plan_chunk[unrecord_id]['modified_start_time']
                item['plan_chunk'] = renew_daily_plan_chunk[unrecord_id]['modified_plan_chunk']
                item['end_time'] = renew_daily_plan_chunk[unrecord_id]['modified_end_time']
            
        final_memory = retrieve_video.memory_integration(memory)
        
        m_list_p = os.path.join(config.MEMORY_PATH,config.SAVE_MEMORY_LIST)
        with open(m_list_p, 'r') as f:
            m_list = json.load(f)
        if p_uid in m_list.keys():
            m_list[p_uid].append("{}_{}".format(final_memory['metadata']['memory_id'], retrieve_chunk_num))
        else:
            m_list[p_uid] = []
            m_list[p_uid].append("{}_{}".format(final_memory['metadata']['memory_id'], retrieve_chunk_num))
        
        with open(m_list_p, 'w') as f:
            json.dump(m_list, f, indent=4)
        with open(memory_path, 'w') as f:
            json.dump(final_memory, f, indent=4)
        
        print("Persona {} has updated the memory {} with {} chunks selected.".format(p_uid,final_memory['metadata']['memory_id'], retrieve_chunk_num))





if __name__ == "__main__":
    # Initialize pipeline
    llg_sim_pip = life_logging_simulation_pip()

    # parameter setting
    with open('generation/ego4d_info/univ_loc_map.json', 'r') as file:
        locations = json.load(file)
    
    # 9 locations
    location = []
    for l in locations["uni_to_loc"].values():
        location.append(l)
  
    # 16 MBTI types
    mbti = [
                "INFP", "INTP", "ISTJ", "ISFJ", 
                "ISTP", "ISFP", "INFJ", "INTJ",
                "ESTP", "ESFP", "ENFP", "ENTP",
                "ESTJ", "ESFJ", "ENFJ", "ENTJ"
            ]
    
    # Matching IoU threshold
    retrieve_threshold = config.IOU_THRESHOLD

    # Lifelog lenth setting
    chunk_num = config.LOG_LENTHS

    for loc_s in location:
        for mbti_s in mbti:
            # generate persona profile
            persona_data = get_persona_profile(loc_s, mbti_s)

            # generate persona daily agenda
            persona_data = get_persona_daily_plan(persona_data)

            # generate daily plan chunks
            persona_data = get_chunked_daily_plan(persona_data)
            cnt  = cnt + 1
            print("persona # {} with {} mbti type located in {} has been created.".format(cnt, mbti_s, loc_s))
            
            end = time.perf_counter()
            print('Persona generation cost: ', (end - start) * 1000)
            
            # Diverse lifelog simulations
            for i in chunk_num:
                start = time.perf_counter()
                llg_sim_pip.life_log_simulation(retrieve_threshold, i, persona_data)
                print("persona # {} simulation with {} chunks selection has been created.".format(cnt, i))
                end = time.perf_counter()
                print('Simulation cost: ', (end - start) * 1000)
                print("-----------------------------")



