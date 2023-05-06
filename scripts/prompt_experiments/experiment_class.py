"""
This module defines the `PromptExperiment` class for running experiments to test different prompt configurations
and hyperparameters with the OpenAI API.

The `PromptExperiment` class is initialized with the following parameters:

- api_key: Your OpenAI API key.
- prompts: A dictionary containing the prompt conditions, where keys are the condition names and values are the
  prompt templates.
- aut_items: A list of items (objects) to be tested with each prompt.
- example_df: A pandas DataFrame containing examples, with columns "prompt" and "response".
- random_seed: (Optional) A random seed for reproducibility. Default is 416.

The `run` method is used to start the experiment, and it requires two parameters:

- n_trials_per_combo: Number of trials per combination of prompt, item, and example count.
- grid_search: A dictionary containing lists of possible values for hyperparameters:
  - temperature
  - frequency_penalty
  - presence_penalty
  - n_examples (only applicable if at least one prompt contains the word "EXAMPLES")
"""

import openai
import json
import multiprocessing
import random
import os
import sys
import logging
import concurrent.futures
from datetime import datetime
from tenacity import RetryCallState
import jsonlines

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log
)  # for exponential backoff


class PromptExperiment:
    def __init__(self, api_key, prompts, aut_items, n_uses, example_df, title="", random_seed=416):
        self.api_key = api_key
        self.prompts = prompts
        self.aut_items = aut_items
        self.n_uses = n_uses
        self.example_df = example_df
        self.random_seed = random_seed
        self.title = title
        random.seed(self.random_seed)

    def handle_prompt(self, args):
        prompt_base, object_name, examples, n_examples, temperature, frequency_penalty, presence_penalty, n_uses = args
        prompt = self.make_prompt(prompt_base, object_name, examples, n_examples, n_uses)
        response = self.generate_responses(prompt, temperature, frequency_penalty, presence_penalty)
        return response

    def make_prompt(self, prompt_base, object_name, examples, n_examples, n_uses):
        prompt = prompt_base.replace("[OBJECT_NAME]", object_name)
        prompt = prompt.replace("[N]", str(n_uses))
        examples = " ".join(['\n- ' + item for item in examples]) + "\n"
        prompt = prompt.replace("[EXAMPLES]", examples)
        #print("PROMPT", prompt)
        return prompt

    @retry(wait=wait_random_exponential(multiplier=30, min=1, max=60), stop=stop_after_attempt(30),
           before_sleep=before_sleep_log(logging, logging.INFO))
    def generate_responses(self, prompt, temperature, frequency_penalty, presence_penalty):
        openai.api_key = self.api_key
        messages = openai.ChatCompletion.create(
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        msg = messages['choices'][0]['message']['content']
        #print(msg)
        return msg

    def get_examples(self, df, prompt, n_examples, seed=416):
        return df[df['prompt'] == prompt].sample(n_examples, random_state=seed)['response'].tolist()

    def run(self, n_trials_per_combo, grid_search):
        now = datetime.now()
        date_string = now.strftime("%Y-%m-%d__%H.%M.%S")
        log_file = f"{self.title}_n{n_trials_per_combo}_{date_string}.log" if self.title else f"experiment_{date_string}.log"
        logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w', format='%(asctime)s %(message)s')
        results_file = f"results_{date_string}.jsonl"
        results_file = f"{self.title}_n{n_trials_per_combo}_{date_string}.jsonl" if self.title else f"results_{date_string}.jsonl"


        logging.info(f"n_trials_combo: {n_trials_per_combo}")

        should_get_examples = any('[EXAMPLES]' in prompt for prompt in self.prompts.values())

        logging.info(f"grid_search parameters: {grid_search}")
        logging.info(f"prompts: {self.prompts}")
        logging.info(f"AUT ITEMS: {self.aut_items}")
        total_requests = len(self.prompts) * len(self.aut_items) * n_trials_per_combo
        logging.info(f"TOTAL REQUESTS: {total_requests}")

        results = []
        condition_counter = 0
        total_counter = 0

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=multiprocessing.cpu_count() - 1) as executor, jsonlines.open(results_file,
                                                                                         mode='w') as outfile:
            for aut_item in self.aut_items:
                for trial in range(n_trials_per_combo):
                    temperature = random.choice(grid_search['temperature'])
                    frequency_penalty = random.choice(grid_search['frequency_penalty'])
                    presence_penalty = random.choice(grid_search['presence_penalty'])

                    if should_get_examples:
                        n_examples = random.choice(grid_search['n_examples'])
                        examples = self.get_examples(self.example_df, aut_item, n_examples, seed=condition_counter)
                    else:
                        examples = []

                    for prompt_name, prompt_base in self.prompts.items():
                        args = (
                            prompt_base,
                            "a " + aut_item,
                            examples,
                            len(examples),
                            temperature,
                            frequency_penalty,
                            presence_penalty,
                            self.n_uses
                        )
                        future = executor.submit(self.handle_prompt, args)
                        generated_response = future.result()

                        row = {
                            'aut_item': aut_item,
                            'prompt_condition': prompt_name,
                            'trial_no': trial,
                            'idx': condition_counter,
                            'examples': examples,
                            'output_responses': generated_response,
                            'n_examples': len(examples),
                            'temperature': temperature,
                            'frequency_penalty': frequency_penalty,
                            'presence_penalty': presence_penalty
                        }
                        results.append(row)
                        total_counter += 1
                        self.random_seed += 1
                        random.seed(self.random_seed)
                        if total_counter % 100 == 0:
                            logging.info(f"{total_counter} of {total_requests}")

                condition_counter += 1

        with jsonlines.open(results_file, mode='w') as writer:
            for row in results:
                writer.write(row)
        logging.info("DONE WITH EXPERIMENT")
