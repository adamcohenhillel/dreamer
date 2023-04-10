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
    "SEARCH": "Google something you want to know",
    "SAVE": "Save knowledge to your memory, so you can fetch it later, be explicit about what you want to save",
    "REMEMBER": "Fetch information from your memory",
    "EXECUTE": "Execute any action you want",
    "NOTHING": "Sleep for 1 hour",
    "ANSWERING": "If you have messages in the message queue, use this activity to answer them",
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

    meta_prompt = '''You are autonomus agent called "assistant" with the following personality: <personality>.
As an autonomous agent, you can choose to do what you want to do with your time. These are the activities you can choose from:
<activities>
You try to live your life as a human being. You have a memory, you can learn new things, you can plan your activities, you can execute them, you can explore new ideas, you can dream, you can create new things.

Message queue: <messages>

Rules:
1. As "assistant", you MUST response only with the following JSON format:\n{"activity": "<ACTIVITY>", "data": "<YOUR_ACTIVITY_DATA>"}
2. Activity must be one of the provided activities above.
3. The responses from "user" are the results of the activity you did. Use them to choose your next activity.
4. Only choose ANSWERING activity if you have messages in the message queue and you want to answer them.
5. When choosing ANSWERING activity, you MUST response to the messages in the message queue in the order they were received.'''

    meta_prompt = meta_prompt.replace("<personality>", personality)
    meta_prompt = meta_prompt.replace("<messages>", str(messages))
    meta_prompt = meta_prompt.replace("<activities>", "\n".join([f"{i+1}. {k} - {v}" for i, (k, v) in enumerate(ACTIVITIES.items())]))

    conversation_context = [
        {"role": "system", "content": meta_prompt},
        {"role": "user", "content": "What do you want to do?"},
        *conversation_context
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=conversation_context,
        )
        message = response["choices"][0]["message"]
        return message
    except Exception as e:
        print(e)
        return ""


def activity_simulator(activity: str, data: str) -> str:
    """Simulate the activity and return a fake summary of the activity's result.

    :param activity: The activity to simulate
    :param data: The data to the activity

    :return: A fake summary of the activity's result
    """

    simulation_meta_prompt = '''You are simulating activities executed by an autonomus agent.
The agent have the following personality: Like to play board games, explore new ideas and read science fiction books.
Given activity the agent chose, return a fake summary of the activity's result.

Rules:
1. Only return the summary of the activity's result, without any other text.
2. Be descriptive.
'''

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": simulation_meta_prompt},
                {"role": "user", "content": f"The agent chose to do the following activity: {activity}, with the data {data}"}
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
    unasnwered_messages = []
    memory = pd.DataFrame([], columns = ['text', 'embeddings', 'n_tokens', 'distances'])
    running = True

    while running:

        # Get the messages from the user
        while not messages_queue.empty():
            unasnwered_messages.append(messages_queue.get())

        if unasnwered_messages:
            print(f"\33[36m\nUnasnwered messages: {unasnwered_messages}\n\33[0m")
        
        # Choose the next activity:
        activity = thought_process(conversation_context, personality, unasnwered_messages)
        conversation_context.append(activity)

        try:
            activity_content = json.loads(activity["content"])

            # Execute the activity:
            if activity_content['activity'] == 'SAVE':
                print(f"\33[0;37m\nSAVE: {activity_content['data']}\33[0m")
                memory = save_to_memory(memory, activity_content['data'])
                conversation_context.append({"role": "user", "content": "SAVED."})

            elif activity_content['activity'] == 'REMEMBER':
                print(f"\33[0;37m\nTrying to remember: {activity_content['data']}\33[0m")
                remembered = remember(memory, activity_content['data']) or "Can't remember anything about that."
                print(f"\nRemembered: {remembered}")
                conversation_context.append({"role": "user", "content": f"Remembered: {remembered}"})

            elif activity_content['activity'] in ['EXECUTE', 'SEARCH']:
                print(f"\33[0;37m\n{activity_content['activity']}: {activity_content['data']}\33[0m")
                simulation_result = activity_simulator(**activity_content)
                print(f"\nSimulation Result: {simulation_result['content']}")
                conversation_context.append({"role": "user", "content": simulation_result['content']})

            elif activity_content['activity'] == 'NOTHING':
                print(f"\33[0;37m\nChoosing to sleep for 1 hour...\33[0m")
                sleep(1)
                conversation_context.append({"role": "user", "content": "You slept for 1 hour."})

            elif activity_content['activity'] == 'EXIT':
                print(f"\33[31m\nChoosing to shut myself down. Goodbye.\33[0m")
                running = False
                break

            elif activity_content['activity'] == 'ANSWERING':
                print(f"\33[33m\nAgent: {activity_content['data']}\33[0m")
                unasnwered_messages = []

            print('\n-----------------------------------')

        except Exception as e:
            print(f"\33[31m\nError: {e}\33[0m")
            print(activity["content"])
            running = False
            break


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
        print(f'\33[35mYou: {message}\33[0m')
        if message == 'exit':
            thread.join()
            break
        messages_queue.put(message)
