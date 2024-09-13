
# import json
# import time
# import google.generativeai as genai

# import os, sys

# from utils import *

# genai.configure(api_key=GOOGLE_API_KEY)
# model = genai.GenerativeModel('gemini-1.5-flash')


# def temp_sleep(seconds=5):
#   time.sleep(seconds)

# def Gemini_single_request(prompt):
#   temp_sleep()

#   response = model.generate_content(prompt)

#   return response.text

# def Gemini_request(prompt):
#   temp_sleep()
#   try:
#     response = model.generate_content(prompt)
#     return response.text
#   except:
#     print(response)
#     return 'gemini ERROR'
  
# def Gemini_safe_generate_response(prompt, 
#                                    example_output,
#                                    special_instruction,
#                                    repeat=3,
#                                    fail_safe_response="error",
#                                    func_validate=None,
#                                    func_clean_up=None,
#                                    verbose=False): 
  
#     prompt = '"""\n' + prompt + '\n"""\n'
#     prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
#     prompt += "Example output json:\n"
#     prompt += '{"output": "' + str(example_output) + '"}'

#     if verbose: 
#         print ("CHAT Gemini PROMPT")
#         print (prompt)

#     for i in range(repeat): 

#         try: 
#             curr_response = Gemini_request(prompt).strip()
#             end_index = curr_response.rfind('}') + 1
#             curr_response = curr_response[:end_index]
#             curr_response = json.loads(curr_response)["output"]

#             if func_validate(curr_response, prompt=prompt): 
#                 return func_clean_up(curr_response, prompt=prompt)
            
#             if verbose: 
#                 print ("---- repeat count: \n", i, curr_response)
#                 print (curr_response)
#                 print ("~~~~")

#         except: 
#             pass

#     return False


# # ============================================================================
# # ###################[SECTION 2: ORIGINAL STRUCTURE] ###################
# # ============================================================================

# def gemini_request(prompt, parameter): 
#   """
#   Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
#   server and returns the response. 
#   ARGS:
#     prompt: a str prompt
#     parameter: a python dictionary with the keys indicating the names of  
#                    the parameter and the values indicating the parameter 
#                    values.   
#   RETURNS: 
#     a str of  response. 
#   """
#   temp_sleep()
#   try:
#     response = model.generate_content(
#                 contents=prompt,
#                 stream=parameter["stream"],
#                 generation_config=genai.GenerationConfig(
#                                   max_output_tokens=parameter["max_tokens"],
#                                   stop_sequences = parameter["stop"],
#                                   top_p=parameter["top_p"],))
#     return response.text
#   except Exception as e: 
#     print (e)
#     return e


# def generate_prompt(curr_input, prompt_lib_file): 
#   """
#   Takes in the current input (e.g. comment that you want to classifiy) and 
#   the path to a prompt file. The prompt file contains the raw str prompt that
#   will be used, which contains the following substr: !<INPUT>! -- this 
#   function replaces this substr with the actual curr_input to produce the 
#   final promopt that will be sent to the GPT3 server. 
#   ARGS:
#     curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
#                 INPUT, THIS CAN BE A LIST.)
#     prompt_lib_file: the path to the promopt file. 
#   RETURNS: 
#     a str prompt that will be sent to OpenAI's GPT server.  
#   """
#   if type(curr_input) == type("string"): 
#     curr_input = [curr_input]
#   curr_input = [str(i) for i in curr_input]
  
#   with open(prompt_lib_file, "r") as f:
#       prompt = f.read()

#   for count, i in enumerate(curr_input):   
#     prompt = prompt.replace(f"!<INPUT {count}>!", i)
#   if "<commentblockmarker>###</commentblockmarker>" in prompt: 
#     prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
#   return prompt.strip()


# def safe_generate_response(prompt, 
#                            parameter,
#                            repeat=5,
#                            fail_safe_response="error",
#                            func_validate=None,
#                            func_clean_up=None,
#                            verbose=True): 
#   if verbose: 
#     print (prompt)

#   for i in range(repeat): 
#     curr_response = gemini_request(prompt, parameter)
#     if verbose: 
#       print ("---- repeat count: ", i, prompt)
#       print (curr_response)
#       print ("~~~~")
#     if func_validate(curr_response,prompt= prompt): 
#       return func_clean_up(curr_response, prompt=prompt)
#   return fail_safe_response


# def get_embedding(text, model="models/text-embedding-004"):
#   temp_sleep()

#   text = text.replace("\n", " ")
#   if not text: 
#     text = "this is blank"
#   return genai.embed_content(
#     model=model,
#     content=text,
#     task_type='SEMANTIC_SIMILARITY')



import json
import time
import google.generativeai as genai
import os
import sys
from queue import Queue
from threading import Thread, Lock
from utils import * 

# Gemini API 키 설정
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# 글로벌 변수
request_queue = Queue()
timestamp_queue = Queue()
rpm_limit = 10
request_interval = 60 / rpm_limit
lock = Lock()

def rate_limited_request():
    while True:
        prompt, parameter, callback = request_queue.get()  # parameter 추가
        current_time = time.time()

        with lock:
            if not timestamp_queue.empty():
                time_diff = current_time - timestamp_queue.queue[0]
                if time_diff < request_interval:
                    time.sleep(request_interval - time_diff)

            retry_count = 0
            while retry_count < 5:  # 최대 5번까지 재시도, 지수 백오프 적용
                try:
                    response = model.generate_content(
                        contents=prompt,
                        stream=parameter.get("stream", False),
                        generation_config=genai.GenerationConfig(
                            max_output_tokens=parameter.get("max_tokens", 256),
                            top_p=parameter.get("top_p", 0.95),
                        )
                    )
                    
                    callback(response.text)
                    break  # 성공 시 루프 종료
                except Exception as e:
                    retry_count += 1
                    sleep_time = min(60, (2 ** retry_count) * request_interval)  # 지수 백오프
                    print(f"{e} 오류 발생, {sleep_time}초 후 재시도...")
                    time.sleep(sleep_time)

            timestamp_queue.put(time.time())
            if timestamp_queue.qsize() > rpm_limit:
                timestamp_queue.get()

        request_queue.task_done()

# 백그라운드 스레드 시작
worker_thread = Thread(target=rate_limited_request, daemon=True)
worker_thread.start()

def Gemini_single_request(prompt, parameter):
    result = []
    request_queue.put((prompt, parameter, lambda x: result.append(x)))
    request_queue.join()
    return result[0]

def Gemini_request(prompt, parameter=None):
    try:
        if parameter is None:
            parameter = {}
        print(parameter)
        return Gemini_single_request(prompt, parameter)
    except Exception as e:
        print('DEBUG : Gemini_request', e)
        time.sleep(10)
        return e


def Gemini_safe_generate_response(prompt,
                                  example_output,
                                  special_instruction,
                                  repeat=3,
                                  fail_safe_response="error",
                                  func_validate=None,
                                  func_clean_up=None,
                                  verbose=False):
    prompt = '"""\n' + prompt + '\n"""\n'
    prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
    prompt += "Example output json:\n"
    prompt += '{"output": "' + str(example_output) + '"}'

    if verbose: 
        print("CHAT Gemini PROMPT")
        print(prompt)

    for i in range(repeat): 
        try: 
            curr_response = Gemini_request(prompt).strip()
            end_index = curr_response.rfind('}') + 1
            curr_response = curr_response[:end_index]
            curr_response = json.loads(curr_response)["output"]

            if func_validate(curr_response, prompt=prompt): 
                return func_clean_up(curr_response, prompt=prompt)
            
            if verbose: 
                print("---- repeat count: \n", i, curr_response)
                print(curr_response)
                print("~~~~")

        except: 
            pass

    return fail_safe_response



def generate_prompt(curr_input, prompt_lib_file):
    if isinstance(curr_input, str):
        curr_input = [curr_input]
    curr_input = [str(i) for i in curr_input]

    with open(prompt_lib_file, "r") as f:
        prompt = f.read()

    for count, i in enumerate(curr_input):
        prompt = prompt.replace(f"!<INPUT {count}>!", i)
    if "<commentblockmarker>###</commentblockmarker>" in prompt:
        prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
    return prompt.strip()

def safe_generate_response(prompt,
                           parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=True):
    if verbose:
        print(prompt)

    for i in range(repeat):
        curr_response = Gemini_request(prompt, parameter)
        if verbose:
            print("---- repeat count: ", i, prompt,'----')
            print(curr_response)
            print("~~~~")
        if func_validate(curr_response, prompt=prompt):
            return func_clean_up(curr_response, prompt=prompt)
    return fail_safe_response

def get_embedding(text, model="models/text-embedding-004"):
    text = text.replace("\n", " ")
    if not text:
        text = "this is blank"
    return genai.embed_content(
        model=model,
        content=text,
        task_type='SEMANTIC_SIMILARITY').values

# 메인 실행 부분 (예시)
if __name__ == "__main__":
    # 예시 사용법
    prompt = "What is the capital of France?"
    response = Gemini_request(prompt)
    print(f"Response: {response}")

    # 안전한 응답 생성 예시
    def dummy_validate(response, prompt):
        return len(response) > 0

    def dummy_clean_up(response, prompt):
        return response.strip()

    safe_response = Gemini_safe_generate_response(
        prompt="Tell me a short joke",
        example_output="Why did the chicken cross the road? To get to the other side!",
        special_instruction="Make sure the joke is family-friendly.",
        func_validate=dummy_validate,
        func_clean_up=dummy_clean_up,
        verbose=True
    )
    print(f"Safe Response: {safe_response}")

    # 임베딩 예시
    embedding = get_embedding("Hello, world!")
    print(f"Embedding: {embedding}")