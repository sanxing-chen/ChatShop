import argparse
import random
import json
from pathlib import Path
from copy import deepcopy
import time

import openai

prompt_assist = [
  {
    "role": "system",
    "content":
      "Your role is to guide users through an online shopping experience, helping them find products that best fit their needs. When a user specifies certain attributes, you analyze these to sift through the available products, based on detailed product descriptions."
  },
]

tools = [
  {
    "type": "function",
    "function": {
      "name": "search",
      "description": "Search with the BM25 search engine. Price can't be searched. This search yields a list of products, each with a unique description and index number.",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "Search query related to the product",
          },
        },
        "required": ["query"],
      },
    }
  },
  {
    "type": "function",
    "function": {
      "name": "select",
      "description": "When the user's criteria clearly match a single product, you finalize your recommendation.",
      "parameters": {
        "type": "object",
        "properties": {
          "index": {
            "type": "number",
            "description": "The number of the product",
          },
        },
        "required": ["index"],
      },
    }
  }
]

question_tools = [
  {
    "type": "function",
    "function": {
      "name": "question",
      "description": "Provide the user with some attributes you find in the database as the user can't see them directly. You can ask questions like question[do you prefer red or black], question[do you like to use it during winter or summer], question[are you allergic to wheat or nuts]",
      "parameters": {
        "type": "object",
        "properties": {
          "question": {
            "type": "string",
            "description": "A question to the user.",
          },
        },
        "required": ["question"],
      },
    }
  },
]

compare_tool = [
  {
    "type": "function",
    "function": {
      "name": "opinion",
      "description": "Ask the user to clarify certain products when more information is needed for a precise decision. You can ask the user's opinion of up to two products, such as opinion[1], opinion[2,3], opinion[3]",
      "parameters": {
        "type": "object",
        "properties": {
          "index": {
            "type": "string",
            "description": "The index of the product(s)",
          },
        },
        "required": ["index"],
      },
    }
  },
]

import backoff

@backoff.on_exception(backoff.expo,
    (
        openai.error.APIError,
        openai.error.Timeout,
        openai.error.RateLimitError,
        openai.error.APIConnectionError,
        openai.error.ServiceUnavailableError
    ),
    max_tries=10)
def get_chat_response(messages, model, temperature=1, n=1, tools=[], tool_choice="auto", max_tokens=1000):
    # count time
    start = time.time()
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        request_timeout=30,
        max_tokens=max_tokens,
        n=n,
        tools=tools,
        tool_choice=tool_choice,
    )
    # print time in seconds (2 decimal places) and flush
    print(f'GPT time: {time.time() - start:.2f}s', flush=True)
    return response

class Agent:
    def __init__(self, model, qmode, cot, strategy='allq', temperature=1, n=5):
        self.model = model
        self.strategy = strategy
        self.qmode = qmode
        self.cot = cot
        self.temperature = temperature
        self.n = n

        if qmode == 'open':
            self.tools = question_tools + tools
        elif qmode == 'compare':
            self.tools = compare_tool + tools
        elif qmode == 'none':
            self.tools = tools
        elif qmode == 'mix':
            self.tools = question_tools + compare_tool + tools

        self.prompt_assist = deepcopy(prompt_assist)
        if qmode == 'none':
            self.prompt_assist[0]['content'] += '\n\n'
        else:
            self.prompt_assist[0]['content'] += " If there are multiple products that match the user's criteria, you ask the user to clarify their preferences before making a final recommendation!\n\n"

        self.reset()

    def reset(self):
        self.messages = deepcopy(self.prompt_assist)
        self.past_actions = []

    def has_searched(self):
        for action in self.past_actions:
            if action.startswith('search'):
                return True
        return False
    
    def last_action(self):
        if len(self.past_actions) == 0:
            return None
        return self.past_actions[-1]
    
    def shrink_messages(self):
        # iterate reversely
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i]['role'] == 'user' and '\n\nProducts:' in self.messages[i]['content']:
                self.messages[i]['content'] = self.messages[i]['content'].split('\n\nProducts:')[0] + '\nDetailed search results deleted.\nThe next action is'

    def take_action(self, message, question_budget):
        action_str = ""
        action_index = None
        question_action_str = self.tools[0]['function']['name']
        if self.strategy == 'random':
            action_str = "action"
            action_index = 0
        elif self.strategy == 'allq':
            action_str = question_action_str if question_budget > 0 and self.has_searched() else "action"
            action_index = 0 if action_str == "action" else 1
        elif self.strategy == 'interleave':
            if question_budget == 0:
                action_str = "action"
                action_index = 0
            elif self.last_action() and self.last_action().startswith('search'):
                action_str = question_action_str
                action_index = 1
            else:
                action_str = "search"
                action_index = 1

        if self.cot:
            self.messages.append({
                "role": "user",
                "content": message + f'\nCould you summarize what you know about the user\'s goal and reason about what your next {action_str} would be? If there are more than three products that match the user\'s criteria, you should gather more information.'
            })
            if action_str == "question":
                self.control_question()

            response = get_chat_response(self.messages, self.model, self.temperature, n=1, tools=self.tools, tool_choice="none", max_tokens=100)
            self.messages.append({
                "role": "assistant",
                "content": response.choices[0]['message']['content']
            })

            self.messages.append({
                "role": "user",
                "content": f'Good! So the next {action_str} is?'
            })

        else:
            self.messages.append({
                "role": "user",
                "content": message + f'\nThe next {action_str} is?'
            })

        if action_str == "question":
            self.control_question()

        # with open('message.txt', 'w') as f:
            # write to plain text file
            # f.write(message)
        # input('Press enter to continue')
        tool_choice = ['auto', {"type": "function", "function": {"name": action_str}}][action_index]
        response = get_chat_response(self.messages, self.model, self.temperature, self.n, self.tools, tool_choice)

        action = ""
        for i in range(len(response.choices)):
            message = response.choices[i]['message']
            if 'tool_calls' not in message and message['content']:
              action = message["content"]
              continue
            try:
              fn = message['tool_calls'][0]['function']
              fn_name = fn['name']
              args = json.loads(fn['arguments'])
              if fn_name == 'search':
                  action = f"search[{args['query']}]"
              elif fn_name == 'select':
                  action = f"select[{str(args['index']).replace('index=', '')}]"
              elif fn_name == 'question':
                  action = f"question[{args['question']}]"
              elif fn_name == 'opinion':
                  action = f"opinion[{str(args['index']).replace('index=', '')}]"
              if action != self.past_actions[-1]:
                break
            except:
              pass
          
        if action and (action.startswith('search') or action.startswith('candidates')):
            self.shrink_messages()

        self.messages.append({
            "role": "assistant",
            "content": action
        })
        self.past_actions.append(action)
        return action

    def control_question(self):
        control_str = f"\n\nNote that the user cannot see the product list and options unless you send description in the question. Be specific, the user can't answer vague questions and those about price or brand."
        for i in range(len(self.messages)):
            self.messages[i]['content'] = self.messages[i]['content'].replace(control_str, '')
        self.messages[-1]['content'] += control_str

if __name__ == '__main__':
    import gymnasium as gym
    from web_agent_site.envs import IPOTextEnv

    args = argparse.ArgumentParser()
    args.add_argument('exp_name', type=str, help='experiment name')

    # Agent arguments
    args.add_argument('--model', type=str,
                        default='gpt-3.5-turbo-1106',
                        choices=['gpt-3.5-turbo', 'gpt-3.5-turbo-1106', 'gpt-4-1106-preview'],
                        help='Agent: select chat model')
    args.add_argument('--strategy', type=str,
                        default='allq',
                        choices=['random', 'allq', 'interleave'],
                        help='Agent: action strategy')
    args.add_argument('--cot', action='store_true',
                        help='Agent: ReAct style prompting')
    
    # IPOEnv arguments
    args.add_argument('--game_mode', type=str,
                        default='full',
                        choices=['full', 'part', 'subj'],
                        help='Env: task ambiguity mode, full is the original WebShop instructions, part is the instructions without salient attributes, subj is the instructions without any attributes.')
    args.add_argument('--qmode', type=str,
                        default='none',
                        choices=['none', 'open', 'compare', 'mix'],
                        help='Env: communication channel, none means no communication between agent and shopper, open enables open-ended communication, compare is the instance-based communication.')
    args.add_argument('--num_question', type=int,
                        default=0,
                        help='Env: maximum number of interactions with the simulated shopper.')
    args = args.parse_args()

    print(args)
    if args.game_mode == 'full':
        game_mode = 0
    elif args.game_mode == 'part':
        game_mode = 1
    elif args.game_mode == 'subj':
        game_mode = 2
    
    env = gym.make('IPOEnv-v0',
                num_products=20,
                num_question=args.num_question,
                game_mode=game_mode)
    start = 500
    num_game = 100
    total_question = 0
    history = []
    rewards = []

    path = Path('chatgpt/logs/' + args.exp_name) / args.model / (f'{args.game_mode}_{args.qmode}' + ('_cot' if args.cot else '') + '_history.json')
    if path.exists():
        with path.open() as f:
            history = json.load(f)
            start += len(history)
            num_game -= len(history)
            for h in history:
                rewards.append(h['info']['scores'][-1])
                total_question += h['info']['question_budget']
        print(f'Loaded {len(history)} games from {path}')

    try:
        for game_idx in range(start, start + num_game):
            print('Game: ', game_idx)
            done = False
            agent = Agent(args.model, args.qmode, args.cot, strategy=args.strategy)
            action = None
            obs, info = env.reset(gid=game_idx)
            print('Observation: ', obs)
            while not done:
                action = agent.take_action(obs, info['question_budget'])
                print('Action: ', action)
                obs, reward, done, _, info = env.step(action)
                print('Observation: ', obs[:100].strip().replace('\n', ' '))
                print('Reward: ', reward)
                print('Done: ', done)
                if done:
                    history.append({
                        'agent': args.model,
                        'agent_messages': agent.messages,
                        'info': info,
                    })
                    rewards.append(reward)
                    total_question += info['question_budget']
                    break
    finally:
        num_game = len(history)
        print(f'Average reward: {sum(rewards) / num_game}')
        print(f'Average number of queries: {total_question / num_game}')

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w') as f:
            json.dump(history, f, indent=2)
        env.close()
