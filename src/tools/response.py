import json
import re
import numpy as np

class Response:
    def __init__(self, response):
        self.response = response
        self.index = 0
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
        
    def save_json(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.response, f, indent=4)
        
    def __repr__(self):
        return f"Response(text={self.text}, score={self.score}, tokens={self.tokens})"
    
    def __str__(self):
        return f"Response(text={self.text}, score={self.score}, tokens={self.tokens})"
    
    def __print__(self):
        return f"Response(text={self.text}, score={self.score}, tokens={self.tokens})"

    
    
    

    