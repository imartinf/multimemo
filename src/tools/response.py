import re
import numpy as np

class Response:
    def __init__(self, response):
        self.response = response
        self.text = self.get_text()
        self.score = self.get_score()
        self.tokens = self.get_tokens()

    def get_text(self):
        if 'choices' in self.response and 'message' in self.response['choices'][0] and 'content' in self.response['choices'][0]['message']:
            return self.response['choices'][0]['message']['content']
        else:
            return 'Error in response'

    def get_score(self):
        s = re.findall(r'\d+\.\d+', self.text)
        score = float(s[0]) if s else -1
        return score

    def get_tokens(self):
        if 'usage' in self.response:
            return self.response['usage']['total_tokens']
        else:
            return 0