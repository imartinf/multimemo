import copy
import re
import time
import numpy as np
import openai

from src.tools.utils import save_json

class ChatGPT:
    """
    ChatGPT is a class that uses the OpenAI API to generate responses to a given prompt.
    """
    def __init__(self, config: dict) -> None:
        """
        Initialize the ChatGPT class.

        :param config: The configuration dictionary.
        :type config: dict

        :return: None
        """

        self.config = config
        self.model_name = config['MODEL']
        self.temperature = config['TEMPERATURE']
        self.max_tokens = 200
        self.top_p = 1
        self.frequency_penalty = 0
        self.presence_penalty = 0
        # self.stop = ['<|im_end|>']
        self.configure()

    def configure(self) -> None:
        """
        Configure the OpenAI API.
        
        :return: None
        """
        openai.api_type = "azure"
        openai.api_key = self.config['OPENAI_API_KEY']
        openai.api_base = self.config['OPENAI_API_BASE']
        openai.api_version = self.config['OPENAI_API_VERSION']

    def get_response(self, prompt: str):
        """
        Get a response from the OpenAI API.

        :param prompt: The prompt to use to generate a response.
        :type prompt: str

        :return: The response from the OpenAI API.
        """
        response = openai.ChatCompletion.create(
            engine=self.model_name,
            messages=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            # stop=self.stop
        )
        return response
    
def exception_handler(e, row, prompt, logger):
    # Print wich row caused the error
    logger.error(row)
    # Print the prompt that caused the error
    logger.error(prompt)
    if isinstance(e, openai.error.APIError): # type: ignore
        # Handle API error
        logger.error("An `APIError` indicates that something went wrong on our side when processing your request. This could be due to a temporary error, a bug, or a system outage.")
        logger.error(e)
        # Wait 10 seconds and go back to the try line
        logger.error("Waiting 10 seconds and trying again...")
        time.sleep(10)
        return True
    elif isinstance(e, openai.error.AuthenticationError): # type: ignore
        # Handle authentication error
        logger.error("An `AuthenticationError` indicates that your API key is missing or invalid.")
        logger.error(e)
        return False
    elif isinstance(e, openai.error.InvalidRequestError): # type: ignore
        # Handle invalid request error
        logger.error("An `InvalidRequestError` indicates that your request is invalid, generally due to invalid parameters.")
        logger.error(e)
        return True
    elif isinstance(e, openai.error.RateLimitError): # type: ignore
        # Handle rate limit error
        logger.error("A `RateLimitError` indicates that you've hit a rate limit.")
        logger.error(e)
        logger.error("Waiting 1 minute to reset the rate limit...")
        time.sleep(60)
        return True
    elif isinstance(e, openai.error.Timeout): # type: ignore
        # Handle timeout error
        logger.error("A `Timeout` indicates that the request timed out.")
        logger.error(e)
        # Wait 10 seconds and go back to the try line
        logger.error("Waiting 10 seconds and trying again...")
        time.sleep(10)
        return True
    elif isinstance(e, openai.error.ServiceUnavailableError): # type: ignore
        # Handle service unavailable error
        logger.error("A `ServiceUnavailableError` indicates that we're experiencing unexpected technical difficulties.")
        logger.error(e)
        # Wait 3 minutes and go back to the try line
        logger.error("Waiting 3 minutes and trying again...")
        time.sleep(180)
        return True
    else:
        # Handle generic OpenAI error
        logger.error("An unexpected error occurred.")
        logger.error(e)
        return False

def get_responses(row, chatgpt, logger, base_prompt):
    responses = []
    for _, caption in enumerate(row['captions']):
        prompt = copy.deepcopy(base_prompt)
        # prompt.append({"role": "user", "content": f"- Caption: {caption}"})
        # prompt.append({"role": "user", "content": f"- Actions: {row['actions']}"})
        prompt[1]['content'] = prompt[1]['content'].format(caption=caption, actions=row['actions'])
        for attempt in range(3):
            try:
                response = chatgpt.get_response(prompt)
            except Exception as e:
                logger.error("Error in response")
                logger.error('Attempt: ' + str(attempt))
                retry = exception_handler(e, row, prompt, logger)
                if retry:
                    continue
                else:
                    response = {'choices': [{'message': {'content': 'Error in response'}}]}
                    break       
            else:
                break
        else:
            response = {'choices': [{'message': {'content': 'Error in response'}}]}
        responses.append(response)
    return responses

def retry_get_responses_if_error_in_response(row, chatgpt, logger, base_prompt):
    responses = []
    recaptions = row['recaptions']
    captions = row['captions']
    for i, recaption in enumerate(recaptions):
        if recaption == 'Error in response' or recaption == '':
            prompt = copy.deepcopy(base_prompt)
            prompt[1]['content'] = prompt[1]['content'].format(caption=captions[i], actions=row['actions'])
            for attempt in range(3):
                try:
                    response = chatgpt.get_response(prompt)
                except Exception as e:
                    logger.error("Error in response")
                    logger.error('Attempt: ' + str(attempt))
                    retry = exception_handler(e, row, prompt, logger)
                    if retry:
                        continue
                    else:
                        response = {'choices': [{'message': {'content': 'Error in response'}}]}
                        break       
                else:
                    break
            else:
                response = {'choices': [{'message': {'content': 'Error in response'}}]}
        else:
            response = row['responses'][i]
        responses.append(response)
    return responses

def get_score_from_response(response):
    s = re.findall(r'\d+\.\d+', response['choices'][0]['message']['content'])
    score = float(s[0]) if s else -1
    return score

def get_text_from_responses(responses):
    texts = []
    for response in responses:
        if 'choices' in response and 'message' in response['choices'][0] and 'content' in response['choices'][0]['message']:
            texts.append(response['choices'][0]['message']['content'])
        else:
            texts.append('Error in response')
    return texts

def get_score_from_responses(responses):
    scores = []
    for response in responses:
        scores.append(get_score_from_response(response))
    return scores

def get_tokens_from_responses(responses):
    tokens = []
    for response in responses:
        if 'usage' in response:
            tokens.append(response['usage']['total_tokens'])
        else:
            tokens.append(0)
    return np.sum(tokens)