import threading
import argparse
from time import sleep
import os
import json
import queue

import openai
from openai.embeddings_utils import distances_from_embeddings
import pandas as pd
import tiktoken


openai.api_key = os.environ.get('OPENAI_APIKEY')
messages_queue = queue.Queue()


ACTIVITIES: dict = {
    "SEARCH": "Google something",
    "SAVE": "Save something to your memory",
    "REMEMBER": "Fetch something from your memory",
    "EXECUTE": "Execute something",
    "NOTHING": "Sleep for 1 hour",
    "ANSWER": "Answer a messages from the message queue",
    "EXIT": "shutdown"
}


def remember(
    memory: pd.DataFrame,
    retrieve: str,
    depth: int = 2
) -> str:
    """Trying to retrieve information from memory

    :param memory: A dataframe of the texts and their embeddings
    :param retrieve: The information to retrieve from memory
    :param depth: How many results to return

    :return: The information retrieved from memory
    """
    info_embedded = openai.Embedding.create(
        input=retrieve,
        engine='text-embedding-ada-002'
    )['data'][0]['embedding']

    # Get the distances from the embeddings
    memory['distances'] = distances_from_embeddings(
        info_embedded,
        memory['embeddings'].values,
        distance_metric='cosine'
    )

    relevant_memories = []
    for _, row in memory.sort_values('distances', ascending=True).head(depth).iterrows():
        relevant_memories.append(row["text"])

    return "\n###\n".join(relevant_memories)


def save_to_memory(
    memory: pd.DataFrame,
    information: str
) -> pd.DataFrame:
    """Saving information to memory

    :param memory: A dataframe of the texts and their embeddings
    :param information: The information to save to memory

    :return: Updated dataframe of the memory with the new information
    """
    # TODO: Should add some randomness to memory failure? so it won't really remember everything
    tokenizer: tiktoken.Encoding = tiktoken.get_encoding("cl100k_base")
    n_tokens = len(tokenizer.encode(information))
    embeddings = openai.Embedding.create(
        input=information,
        engine='text-embedding-ada-002'
    )['data'][0]['embedding']
    new_memory = pd.concat([
        memory,
        pd.DataFrame([{'text': information, 'n_tokens': n_tokens, 'embeddings': embeddings}])
    ], ignore_index=True)
    return new_memory


def thought_process(
    conversation_context: list,
    personality: str,
    messages: list
):
    """Uses to decide what to do next, based on the previous activities and the constraints of the agent's world.
    """

    meta_prompt = '''You are autonomus agent called "assistant" with the following personality: {personality}.
As an autonomous agent, you can choose to do what you want to do with your time. These are the activities you can choose from:
{activities}
You try to live your life as a human being. You have a memory, you can learn new things, you can plan your activities, you can execute them, you can explore new ideas, you can dream, you can create new things.
You are also a social being. You can communicate with other agents, you can ask them for help, you can ask them for advice, you can ask them for information, you can ask them for resources, you can ask them for money, you can ask them for a favor, you can ask them for a date, you can ask them for a job, you can ask them for a loan, you can ask them for a gift, you can

Message queue: {messages}
Rules:
1. As "assistant", you MUST response only with the following JSON format:\n{"activity": "<ACTIVITY>", "input": "<YOUR_INPUT>"}
2. Activity must be one of the provided activities above.
3. The responses from "user" are the results of the activity you did. Use them to choose your next activity.
'''
    meta_prompt = meta_prompt.replace("{personality}", personality)
    meta_prompt = meta_prompt.replace("{messages}", str(messages))
    meta_prompt = meta_prompt.replace("{activities}", "\n".join([f"{i+1}. {k} - {v}" for i, (k, v) in enumerate(ACTIVITIES.items())]))

    conversation_context = [
        {"role": "system", "content": meta_prompt},
        {"role": "user", "content": "What do you want to do?"},
        *conversation_context
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
        )
        message = response["choices"][0]["message"]
        return message
    except Exception as e:
        print(e)
        return ""


def activity_simulator(activity: str, input: str) -> str:
    """Simulate the activity and return a fake summary of the activity's result.

    :param activity: The activity to simulate
    :param input: The input to the activity

    :return: A fake summary of the activity's result
    """
    simulation_meta_prompt = '''You are simulating activities executed by an autonomus agent.
The agent have the following personality: Like to play board games, explore new ideas and read science fiction books.
Given activity the agent chose, return a fake summary of the activity's result.
    '''

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": simulation_meta_prompt},
                {"role": "user", "content": f"The agent chose to do the following activity: {activity}, with the input {input}"}
            ],
        )
        message = response["choices"][0]["message"]
        return message
    except Exception as e:
        print(e)
        return ""


def agent_loop(personality: str):
    """The main loop of the agent

    :param personality: The personality of the agent
    """
    print(f'\33[36m\nBooting...\33[0m')

    conversation_context = []
    memory = pd.DataFrame([], columns = ['text', 'embeddings', 'n_tokens', 'distances'])
    active = True

    while active:

        new_messages = []
        while not messages_queue.empty():
            new_messages = messages_queue.get()
            new_messages.append(message)

        activity = thought_process(conversation_context, personality, new_messages)
        conversation_context.append(activity)

        activity_content = json.loads(activity["content"])
        print(f"Chosen activity: {activity_content['activity']}, with input: {activity_content['input']}")

        if activity_content['activity'] == 'SAVE':
            memory = save_to_memory(memory, activity_content['input'])
            conversation_context.append({"role": "user", "content": "SAVED."})

        elif activity_content['activity'] == 'REMEMBER':
            remembered = remember(memory, activity_content['input']) or "Can't remember anything about that."
            print(f"Remembered: {remembered}")
            conversation_context.append({"role": "user", "content": f"Remembered: {remembered}"})

        elif activity_content['activity'] in ['EXECUTE', 'SEARCH']:
            simulation_result = activity_simulator(**activity_content)
            print(f"Simulation Result: {simulation_result['content']}")
            conversation_context.append({"role": "user", "content": simulation_result['content']})

        elif activity_content['activity'] == 'NOTHING':
            print("Sleeping for 1 hour...")
            sleep(1)
            conversation_context.append({"role": "user", "content": "You slept for 1 hour."})

        elif activity_content['activity'] == 'EXIT':
            print("Shutting down...")
            active = False
            break

        elif activity_content['activity'] == 'ANSWER':
            print("Dreaming...")
            sleep(1)
            conversation_context.append({"role": "user", "content": "You dreamed for 1 hour."})
        
        print('************************')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A dreaming Agent")
    parser.add_argument("--personality", action="store", dest="personality", help="The task to do")
    args = parser.parse_args()
    
    if not args.personality:
        raise SystemExit("--personality must be provided.")


    # running agent:
    thread = threading.Thread(target=agent_loop, args=(args.personality, ))
    thread.start()

    # user interface loop to send messages to agent:
    while True:
        message = input(" ")
        print(f'\33[35m\You asked: {message}\33[0m')
        if message == 'exit':
            thread.join()
            break
        q.put(message)
