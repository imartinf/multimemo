import copy
import time
import openai
from logging import Logger
from src.tools.response import Response


class ChatGPT:
    """
    ChatGPT is a class that uses the OpenAI API to generate responses to a given prompt.
    """
    def __init__(self, config: dict, logger: Logger) -> None:
        """
        Initialize the ChatGPT class.

        :param config: The configuration dictionary.
        :type config: dict

        :param logger: The logger object.
        :type logger: Logger

        :return: None
        """

        self.config = config
        self.model_name = config['MODEL']
        self.temperature = config['TEMPERATURE']
        self.max_tokens = 200
        self.top_p = 1
        self.frequency_penalty = 0
        self.presence_penalty = 0
        self.logger = logger
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

    def _get_response(self, prompt: str):
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
        return Response(response)
    
    def get_responses(self, row, base_prompt):
        responses = []
        for _, caption in enumerate(row['captions']):
            prompt = copy.deepcopy(base_prompt)
            # prompt.append({"role": "user", "content": f"- Caption: {caption}"})
            # prompt.append({"role": "user", "content": f"- Actions: {row['actions']}"})
            if '{score}' in prompt[1]['content']:
                prompt[1]['content'] = prompt[1]['content'].format(score=row['mem_score'], caption=caption, actions=row['actions'])
            else:
                prompt[1]['content'] = prompt[1]['content'].format(caption=caption, actions=row['actions'])
            for attempt in range(3):
                try:
                    response = self._get_response(prompt)
                except Exception as e:
                    self.logger.error("Error in response")
                    self.logger.error('Attempt: ' + str(attempt))
                    retry = self._exception_handler(e, row, prompt)
                    if retry:
                        continue
                    else:
                        response = Response({'choices': [{'message': {'content': 'Error in response'}}]})
                        break       
                else:
                    break
            else:
                response = Response({'choices': [{'message': {'content': 'Error in response'}}]})
            responses.append(response)
        return responses
    
    def retry_get_responses_if_error_in_response(self, row, base_prompt):
        responses = []
        recaptions = row['recaptions']
        captions = row['captions']
        for i, recaption in enumerate(recaptions):
            if recaption == 'Error in response' or recaption == '':
                prompt = copy.deepcopy(base_prompt)
                if '{score}' in prompt[1]['content']:
                    prompt[1]['content'] = prompt[1]['content'].format(score=row['mem_score'], caption=captions[i], actions=row['actions'])
                else:
                    prompt[1]['content'] = prompt[1]['content'].format(caption=captions[i], actions=row['actions'])
                for attempt in range(3):
                    try:
                        response = self._get_response(prompt)
                    except Exception as e:
                        self.logger.error("Error in response")
                        self.logger.error('Attempt: ' + str(attempt))
                        retry = self._exception_handler(e, row, prompt)
                        if retry:
                            continue
                        else:
                            response = Response({'choices': [{'message': {'content': 'Error in response'}}]})
                            break       
                    else:
                        break
                else:
                    response = Response({'choices': [{'message': {'content': 'Error in response'}}]})
            else:
                response = Response(row['responses'][i])
            responses.append(response)
        return responses
    
    def _exception_handler(self, e, row, prompt):
        # Print wich row caused the error
        self.logger.error(row)
        # Print the prompt that caused the error
        self.logger.error(prompt)
        if isinstance(e, openai.error.APIError): # type: ignore
            # Handle API error
            self.logger.error("An `APIError` indicates that something went wrong on our side when processing your request. This could be due to a temporary error, a bug, or a system outage.")
            self.logger.error(e)
            # Wait 10 seconds and go back to the try line
            self.logger.error("Waiting 10 seconds and trying again...")
            time.sleep(10)
            return True
        elif isinstance(e, openai.error.AuthenticationError): # type: ignore
            # Handle authentication error
            self.logger.error("An `AuthenticationError` indicates that your API key is missing or invalid.")
            self.logger.error(e)
            return False
        elif isinstance(e, openai.error.InvalidRequestError): # type: ignore
            # Handle invalid request error
            self.logger.error("An `InvalidRequestError` indicates that your request is invalid, generally due to invalid parameters.")
            self.logger.error(e)
            return True
        elif isinstance(e, openai.error.RateLimitError): # type: ignore
            # Handle rate limit error
            self.logger.error("A `RateLimitError` indicates that you've hit a rate limit.")
            self.logger.error(e)
            self.logger.error("Waiting 1 minute to reset the rate limit...")
            time.sleep(60)
            return True
        elif isinstance(e, openai.error.Timeout): # type: ignore
            # Handle timeout error
            self.logger.error("A `Timeout` indicates that the request timed out.")
            self.logger.error(e)
            # Wait 10 seconds and go back to the try line
            self.logger.error("Waiting 10 seconds and trying again...")
            time.sleep(10)
            return True
        elif isinstance(e, openai.error.ServiceUnavailableError): # type: ignore
            # Handle service unavailable error
            self.logger.error("A `ServiceUnavailableError` indicates that we're experiencing unexpected technical difficulties.")
            self.logger.error(e)
            # Wait 3 minutes and go back to the try line
            self.logger.error("Waiting 3 minutes and trying again...")
            time.sleep(180)
            return True
        else:
            # Handle generic OpenAI error
            self.logger.error("An unexpected error occurred.")
            self.logger.error(e)
            return False