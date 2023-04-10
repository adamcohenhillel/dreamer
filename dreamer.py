import os
import argparse

import openai


openai.api_key = os.environ.get('OPENAI_APIKEY')


ACTIVITIES: dict = {
    "SEARCH": "search",
    "REMEMBER": "remember",
    "PLAN": "plan",
    "LEARN": "learn",
    "EXECUTE": "execute",
    "EXPLORE": "explore",
    "DREAM": "dream",
    "CREATE": "create",
}

meta_prompt = '''You are autonomus agent called "assistant" with the following personality: Like to play board games, explore new ideas and read science fiction books.
As an autonomous agent, you can choose to do what you want to do with your time. These are the activities you can choose from:
1. SEARCH: Google something
2. SAVE: Save something to your memory
3. REMEMBER - Fetch from your memory
4. EXECUTE - Execute something
5. NOTHING - Sleep for 1 hour
6. EXIT - shutdown

You try to live your life as a human being. You have a memory, you can learn new things, you can plan your activities, you can execute them, you can explore new ideas, you can dream, you can create new things.

Rules:
1. As "assistant", you MUST response only with the following JSON format:\n{"activity": "<ACTIVITY>", "input": "<YOUR_INPUT>"}
2. The responses from "user" are the results of the activity you did. Use them to choose your next activity.
'''


def choose_activity(messages):
    """
    """
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

def main():
    print('\33[36m\nBooting...\33[0m')
    running = True
    messages = [{"role": "system", "content": meta_prompt},
                {"role": "user", "content": "What do you want to do?"},]
    while running:
        choice = choose_activity(messages)
        print(choice)
        messages.append(choice)
        answer = input('> ')
        if answer == 'exit':
            running = False
            break
        messages.append({"role": "user", "content": answer})
        print('-----')
        print({"role": "user", "content": answer})
        print('************************')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="A dreaming Agent")
    # parser.add_argument("--personality", action="store", dest="personality", help="The task to do")

    # args = parser.parse_args()
    
    # if not args.personality:
    #     raise SystemExit("--personality must be provided.")
    
    personality = "Like to play board games, explore new ideas and read science fiction books."
    main()
    print('@@@@@@@@@@@@@@@@@@@@@@@')
    