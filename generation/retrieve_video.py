import json
import statistics
import uuid
from sentence_transformers import SentenceTransformer
import chromadb
import random
from utils import generate_response_multiple
from datetime import datetime, timedelta
import generation.config as config
import os
import copy
os.environ['CUDA_VISIBLE_DEVICES'] = '5'


def evenly_sample_chunks(plan_chunks, num_samples):
    if len(plan_chunks)<num_samples:
        return plan_chunks
    # Calculate chunk size (round up if not evenly divisible)
    total_chunks = len(plan_chunks)
    interval = total_chunks // num_samples
    left = total_chunks % num_samples
    interval_list = [interval for i in range(num_samples)]
    idx = random.sample(range(num_samples),left)
    for choose in idx:
        interval_list[choose] = interval_list[choose] + 1
    start_list = [sum(interval_list[:i]) for i in range(num_samples)]

    # Sample at intervals
    sampled_chunks = []
    for i in range(num_samples):
        start = start_list[i]
        end = start + interval_list[i]
        if i == num_samples - 1:
            end = total_chunks
        chunk = plan_chunks[start:end]
        sampled_chunks.append(random.choice(chunk))

    # Ensure exactly 'num_samples' samples by truncating or adding the last item if necessary
    if len(sampled_chunks) > num_samples:
        indices = sorted(random.sample(range(len(sampled_chunks)), num_samples))
        sampled_chunks = [sampled_chunks[i] for i in indices]
    elif len(sampled_chunks) < num_samples and total_chunks > 0:
        sampled_chunks.append(plan_chunks[-1])
    
    return sampled_chunks


def classify_time(start_time_str):
    # Define time ranges for classification
    day_start = datetime.strptime("06:00", "%H:%M").time()
    twilight_start = datetime.strptime("17:00", "%H:%M").time()
    night_start = datetime.strptime("19:00", "%H:%M").time()
    # Convert the time string to a datetime.time object
    s = datetime.strptime(start_time_str, "%H:%M").time()
    
    # Classify based on the time ranges
    if s < twilight_start and s > day_start:
        return 'daytime'
    elif twilight_start <= s < night_start:
        return 'twilight'
    else:
        return 'nighttime'

def filter_uids(ref_info, video_uids, iou_thr, collection, sum_data, map_uid):
    # Query with metadata filtering
    filtered_results = collection.get(
        where={"$and": [
            {"video_uid": {"$nin": video_uids}},
            {"video_source": ref_info['global_location']},
            {"main_scene": {"$in": [ref_info['location'],'mixed']}},
            {"time_period": {"$in": [ref_info['time_period'],'not know']}}
        ]
        }
    )

    # Extract metadata, and video_scenarios from filtered results
    metadatas = filtered_results['metadatas']
    filtered_uids = []
    for metadata in metadatas:
        scenarios = sum_data[map_uid[metadata['video_uid']]]['video_scenarios']
        iou = calculate_scenarios_iou(ref_info['matched_scenarios'], scenarios)
        if iou >= iou_thr:  # Only keep those with IoU >= iou_thr
            filtered_uids.append(metadata['video_uid'])

    # If no matching, lower the threshold
    if len(filtered_uids) == 0:
        for metadata in metadatas:
            scenarios = sum_data[map_uid[metadata['video_uid']]]['video_scenarios']
            iou = calculate_scenarios_iou(ref_info['matched_scenarios'], scenarios)
            if iou >= 0.2:  # Lower the IoU threshold>= 0.2
                filtered_uids.append(metadata['video_uid'])

    # If still no, remove the location constraints
    if len(filtered_uids) == 0:
        filtered_results = collection.get(
            where={"$and": [
                {"video_uid": {"$nin": video_uids}},
                {"main_scene": {"$in": [ref_info['location'],'mixed']}},
                {"time_period": {"$in": [ref_info['time_period'],'not know']}}
            ]
            }
        )
        metadatas = filtered_results['metadatas']
        for metadata in metadatas:
            scenarios = sum_data[map_uid[metadata['video_uid']]]['video_scenarios']
            iou = calculate_scenarios_iou(ref_info['matched_scenarios'], scenarios)
            if iou >= 0.2:  # Lower the IoU threshold>= 0.2
                filtered_uids.append(metadata['video_uid'])
        if len(filtered_uids) == 0:
            for metadata in metadatas:
                filtered_uids.append(metadata['video_uid'])

    return filtered_uids

# Helper function to calculate IoU of scenarios
def calculate_scenarios_iou(list1, list2):
    set1, set2 = set(list1), set(list2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if len(union) == 0: 
        return 0.0
    return len(intersection) / len(union)

def reflection(template_path, persona_info, renewed_plan, reocrds_idx, gen_tool):
    # Construct prompt
    f = open(template_path, "r")
    prompt = f.read()
    personality_traits = persona_info.get("personality_traits", {}).get("character_traits", [])
    personality_traits = ', '.join(personality_traits)
    lifestyle = persona_info.get("lifestyle", "")
    recorded_daily_plan_chunk = "\n".join([str(item) for item in renewed_plan[:reocrds_idx+1]])
    unrecorded_daily_plan_chunk = "\n".join([str(item) for item in renewed_plan[reocrds_idx+1:]])
    hobbies = persona_info.get("hobbies", [])
    hobbies = ', '.join(hobbies)

    prompt = prompt.replace(f"<personality_traits>", f"{personality_traits}")
    prompt = prompt.replace(f"<lifestyle>", lifestyle)
    prompt = prompt.replace(f"<recorded_daily_plan_chunk>", f"{recorded_daily_plan_chunk}")
    prompt = prompt.replace(f"<unrecorded_daily_plan_chunk>", f"{unrecorded_daily_plan_chunk}")
    prompt = prompt.replace(f"<hobbies>", f"{hobbies}")

    # Get reflection
    response =  gen_tool.gen_response(prompt)
    response = response.split("```json")[1]
    response = response.split("```")[0]
    response = response.replace("```", "")
    response = response.replace("json", "")
    raw_json_object = json.loads(response)

    # Ensure reasonable reflection
    if len(renewed_plan[reocrds_idx+1:]) != len(raw_json_object):
        return reflection(template_path, persona_info, renewed_plan, reocrds_idx, gen_tool)
    check_s = datetime.strptime(raw_json_object[0]['modified_start_time'], "%H:%M")
    check_e = datetime.strptime(renewed_plan[reocrds_idx]['record_end_time'], "%H:%M")
    delta = timedelta(seconds=150)
    limit = check_e + delta
    wee_hrs = datetime.strptime("03:00", "%H:%M")
    if check_s<wee_hrs:
        check_s = check_s + timedelta(hours=24)
    if check_s < limit:
        return reflection(template_path, persona_info, renewed_plan, reocrds_idx, gen_tool)
    
    return raw_json_object


def generate_ref_info(template_path, plan_chunk_info, ref_info, gen_tool):
    # Generate reference info of each input plan chunk
    f = open(template_path, "r")
    prompt = f.read()
    start_time = plan_chunk_info["start_time"]
    plan_chunk = plan_chunk_info["plan_chunk"]
    end_time= plan_chunk_info["end_time"]
    scenarios_list = ref_info

    prompt = prompt.replace("<start_time>", f"{start_time}")
    prompt = prompt.replace("<plan_chunk>", f"{plan_chunk}")
    prompt = prompt.replace("<end_time>", f"{end_time}")
    prompt = prompt.replace("<scenarios>", f"{scenarios_list}")
    response = gen_tool.gen_response(prompt)
    response = response.replace("```", "")
    response = response.replace("json", "")
    raw_json_object = json.loads(response)

    # Avoid error/empty scenarios
    none_flag = 0
    for item in raw_json_object['matched_scenarios']:
        if item not in scenarios_list:
            none_flag = 1
    if len(raw_json_object['matched_scenarios'])==0:
        return generate_ref_info(template_path, plan_chunk_info, ref_info, gen_tool)
    if none_flag==1:
        return generate_ref_info(template_path, plan_chunk_info, ref_info, gen_tool)
            
    return raw_json_object


def update_memory(plan_chunk_info):
    memory_chunk = {}

    start_time = plan_chunk_info["start_time"]
    act_activities = plan_chunk_info["selected_video_sum"]
    duration_sec = plan_chunk_info["selected_video_duration"]
    start_time_obj = datetime.strptime(start_time, "%H:%M")
    end_time_obj = start_time_obj + timedelta(seconds=duration_sec)
    end_time = end_time_obj.strftime("%H:%M")

    memory_chunk["start_time"] = start_time
    memory_chunk["retrieved_video"] = plan_chunk_info["selected_video_uid"][0][0]
    memory_chunk["actual_activities"] = act_activities
    memory_chunk["end_time"] = end_time
    return memory_chunk

def memory_integration(memory):
    integration = {}
    integration['metadata'] = memory[0]
    integration['memory_content'] = []
    statistics = {}

    retrieved = {}
    for item in memory[1:]:
        retrieved["video_uid"] = item["video_uid"]
        retrieved["start_time"] = item["retrieval_time"]
        retrieved["end_time"] = item["retrieval_end_time"]
        retrieved["duration"] = item["video_duration"]
        integration['memory_content'].append(retrieved)

    total_dur = 0
    selected_videos = []
    for idx, item in enumerate(integration['memory_content']):
        total_dur = total_dur + item["duration"]
        selected_videos.append(item["video_uid"])
    statistics["total_duration"] = total_dur
    statistics["selected_videos"] = selected_videos
    integration["statistics"] = statistics
    

