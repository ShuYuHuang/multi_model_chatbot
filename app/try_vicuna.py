import requests
from time import time
print(requests.__file__)
# VICUNA_URL = 'http://10.100.100.104:8013/v1/completions'
# VICUNA_URL = 'http://10.100.100.104:8014/v1/completions'
# VICUNA_URL = 'http://10.100.100.106:8014/v1/completions'
# VICUNA_URL = 'http://10.100.100.104:8013/v1/chat/completions'
# VICUNA_URL = 'http://10.100.100.104:8014/v1/chat/completions'
VICUNA_URL = 'http://10.100.100.106:8014/v1/chat/completions'
VICUNA_HEADER = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}
prompts = ['Write me a python script that calculate 5+5',
    'Write me a python script that print "Hello World"' ,
    'Write me a poam about Paris',
    'What is the biggist country?',
    'What is the capital of UAE? give me simple answer']
# prompts = ['Write me a python script that calculate 5+5']
           
# prompts = [
#     """Determine the characters for the transcript in SRT format, and assign the character to the beginning of transcript, e.g. \n99\n00:32:01.000 --> 00 00:32:02.000\nSpeaker C: The suspect didn't show up last night\n\nHere is the input:\n0\n00:00:00.000 --> 00:00:03.200\n Doesn't really look like anyone's been doing cocaine off that table, does it?\n\n1\n00:00:05.600 --> 00:00:07.900\n With all due respect, I'm not sure you know how that works.\n\n2\n00:00:08.800 --> 00:00:10.200\n I'm asking if you do.\n\n3\n00:00:10.800 --> 00:00:12.200\n You've testified you've done cocaine.\n\n4\n00:00:12.200 --> 00:00:12.800\n I have.\n\n5\n00:00:13.300 --> 00:00:17.400\n Doesn't really look like Mr. Depp or anyone was doing cocaine off that table, does it?\n\n6\n00:00:17.600 --> 00:00:20.100\n Uh, I beg to differ with you on that.\n\n7\n00:00:20.100 --> 00:00:23.000\n When you snort cocaine, typically it goes into your nose.\n\n\n\n\n"""
# ]


for prompt in prompts:
    json_data = {
                # "model": "vicuna-13b-v1.5-16k",
                "model": "gpt-3.5-turbo",
                "max_tokens": 512,
                # "prompt": f"\nRequest:\n {prompt} \nResponse:\n",
                # "messages": [
                #     {"role": "user", "content": prompt}
                # ],
                "messages": [
                    {"role": "user", "content": f"\nRequest:\n {prompt} \nResponse:\n"}
                ],
                # "prompt": prompt,
                "temperature": 0.0,
                "stream": "false"
            }

    print("start")
    start_time = time()
    response = requests.post(
                VICUNA_URL,
                headers=VICUNA_HEADER,
                json=json_data)

    # print (response.json()['choices'][0]['text'])
    print (response.json()['choices'][0]['message']['content'])
    print(f"\n---{time()- start_time} seconds")
    
    # for res in response:
    #     print(f"\n---{time()- start_time} seconds")
    #     # print(res.decode("utf-8").json()['choices'][0]['text'])
        
        