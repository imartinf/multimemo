import openai

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
    
