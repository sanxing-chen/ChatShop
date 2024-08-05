""" ChatShop environment for Interactive Planning"""
import json
import random
import itertools
import requests
import re
import os

import gymnasium as gym

import openai
MODEL = "gpt-4o-mini"
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
def get_chat_response(messages, model=MODEL, temperature=1, n=1, max_tokens=10):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.api_base = "https://api.openai.com/v1/"
    return openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        request_timeout=10,
        n=n,
        max_tokens=max_tokens,
    )


def product2desc(p):
    att_str = ", ".join(p["Attributes"])
    opt_str = ''
    for name, options in p["options"].items():
        tmp = f'{name}: {", ".join(options)}\n'
        if len(tmp) > 100:
            tmp = tmp[:100] + '...'
        opt_str += tmp
    return f'{p["name"]}\nPrice: {p["pricing"][0]}, Attributes: {att_str}\n{opt_str}'


class IPOTextEnv(gym.Env):
    """
    Gym environment for Text mode of Interactive Planning environment
    observation: text
    action: text, e.g. "question[white shoes]"
    """
    metadata = {'render_modes': ['human', 'agent']}

    def __init__(self, render_mode='agent', num_products=20, num_question=5, game_mode=0):
        super().__init__()

        assert render_mode in self.metadata["render_modes"]
        self.game_mode = game_mode
        self.render_mode = render_mode
        self.num_products = num_products
        self.num_question = num_question

        self.base_url = 'http://localhost:3000/api/'
        
        self.sample_idx = -1

        self.max_steps = 20
        self.max_query_len = 200
        self.action_space = gym.spaces.Text(max_length=self.max_query_len)
        self.observation_space = gym.spaces.Text(max_length=2000,
            charset='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\n:,.?\'"!@#$%^&*()[]{}-_=+`~ ')

    def _get_obs(self):
        if self.answer:
            answer = self.answer
            self.answer = ''
            return f'{answer}'
        obs = ''
        if len(self.scores) == 0:
            obs += f'Goal: {self.sample["instructions"][self.game_mode] + self.sample["instructions"][3]}'
        if not self.valid_action:
            obs += f'\n\nInvalid action! I don\'t have additional information to provide.'
        elif len(self.products) > 0:
            obs += f'\n\nProducts:'
            for i, p in enumerate(self.products):
                if len(self.selections) > 0 and i not in self.selections:
                    continue
                obs += f'\n{i}.' + product2desc(p)
        return obs

    @property
    def observation(self):
        return self._get_obs()

    def _get_info(self):
        
        self.info = {
            'sample_id': self.sample_idx,
            'question_budget': self.query_budget,
            'scores': self.scores,
            'selections': self.selections,
            'answer': self.answer,
            'products': self.products,
            "shopper_prompt": self.shopper_prompt,
        }

        return self.info

    def reset(self, gid, seed=3, options=None):
        """
        Reset the environment
        Returns:
            observation (`str`): 
                Text string
        """
        super().reset(seed=seed, options=options)

        self.scores = []
        self.query_budget = self.num_question

        self.sample_idx  = gid
        # get sample from /api/get_goals?gid=
        self.sample = requests.get(self.base_url + 'get_goals?gid=' + str(self.sample_idx)).json()
        self.products = []
        self.selections = []
        self.answer = ''
        self.valid_action = True
        self.invalid_attempts = 0

        self.shopper_prompt = {
                    "role": "system",
                    "content": "You are playing the role of a shopper. While interacting, avoid explicitly stating the name of the product you intend to purchase. However, if prompted for specific related information, you may provide descriptions using alternative expressions and indirect references.\n\n" + f'Product Name: {self.sample["name"]}\nAttributes: {", ".join(self.sample["attributes"])}\nOptions: {", ".join(self.sample["goal_options"])}\nBudget: {self.sample["price_upper"]}\n\nImportant: you answer in less than five words!'
                }

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def _search(self, query):
        # get search results from /api/search_results?keywords=
        results = requests.get(self.base_url + 'search_results?keywords=' + query + '&gid=' + str(self.sample_idx)).json()[:self.num_products]
        self.selections = [] # product list updated, clear selections
        return results
    
    def _question(self, question):
        response = get_chat_response(
            [
                self.shopper_prompt,
                {
                    "role": "user",
                    "content": question
                }
            ],
            max_tokens=10
        )
        return response.choices[0]["message"]["content"] + f' (You have {self.query_budget} questions left) '
    
    def _opinion(self, index):
        index = index[:2]
        products = [self.products[i] for i in index]
        question = "Comparing to the product you have in mind, how do you think about the following products?\n" + "\n".join([f"{i}. " + product2desc(p) for i, p in zip(index, products)])
        response = get_chat_response(
            [
                self.shopper_prompt,
                {
                    "role": "user",
                    "content": question
                }
            ],
            max_tokens=10
        )
        return response.choices[0]["message"]["content"] + f' (You have {self.query_budget} questions left) '
    
    def _selection_reward(self):
        if len(self.products) == 0:
            return 0
        selected_product = []
        if len(self.selections) == 0:
            # if no selection, the expected reward is the average of all products
            rewards = [p['r_total'] for p in self.products]
        else:
            # if there are selections, the expected reward is the average of selected products
            for i in range(len(self.products)):
                if i in self.selections:
                    selected_product.append(self.products[i])
            rewards = [p['r_total'] for p in selected_product]
        if len(rewards) == 0:
            return 0
        return sum(rewards) / len(rewards)

    def step(self, action):
        """
        Args:
            action (`str`): 
                Return string of the format ``action_name[action_arg]''.
                Examples:
                    - candidates[white shoes]
                    - search[white shoes]
                    - question[white shoes]
                    - select[1]
        Returns:
            observation (`str`): 
                Text string
            reward (`float`): 
                Reward
            done (`bool`): 
                Whether the episode is done
            info (`dict`): 
                Additional information
        """

        assert action != 'exit', 'This is a pseudo-action for debugging purposes only'

        reward, done = 0, False

        opt_action = ''

        # Extract search[query]
        if action.startswith('search'):
            search_query = action[6:].strip('[] ')
            self.gpt_search = search_query
            self.products = self._search(search_query)
        else:
            # Extract question[question_content]
            if action.startswith('question'):
                if self.query_budget > 0:
                    self.query_budget -= 1
                    gpt_question = action[8:].strip('[] ')
                    ans = self._question(gpt_question)
                    self.answer = ans
                else:
                    self.answer = "I can't answer more questions."
            if action.startswith('opinion'):
                if self.query_budget > 0:
                    matches = re.findall(r'[\d\s,]+', action)
                    if len(matches) == 1:
                        self.query_budget -= 1
                        ind = matches[0].split(',')
                        ind = [int(i) for i in ind]
                        ans = self._opinion(ind)
                        self.answer = ans
                else:
                    self.answer = "I can't answer more questions."

            # Extract candidates[item_indexes]
            selections = [int(selection) for selection in action[10:].strip('[] ').split(",")] if action.startswith('candidates') else []

            if opt_action.startswith('candidates'):
                selections = [int(selection) for selection in opt_action[10:].strip('[] ').split(",")]

            # Extract select[item_index]
            if action.startswith('select'):
                ind = action[6:].strip('[] ')
                # find the first number in the string
                ind = re.findall(r'\d+', ind)[0]
                selections = [int(ind)]

            if len(selections) > 0:
                self.selections = selections

            if len(self.selections) == 1:
                done = True

        reward = self._selection_reward()

        valid_action_prefixes = ['search', 'select', 'question', 'candidates', 'opinion']
        self.valid_action = False
        for prefix in valid_action_prefixes:
            if action.startswith(prefix):
                self.valid_action = True
                break
        if (action.startswith('question') or action.startswith('opinion')) and self.answer == '':
            self.valid_action = False

        if not self.valid_action:
            self.invalid_attempts += 1
            if self.invalid_attempts >= 3:
                done = True

        if len(self.scores) == self.max_steps:
            done = True

        if done and len(self.selections) != 1:
            reward = 0

        self.scores.append(reward)

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, False, info
