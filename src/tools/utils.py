import json
import os


def save_json(json_data, path2save, json_indent):
    messages_JSON = json.dumps(json_data, indent=json_indent)
    # Create folder if it doesn't exist
    os.makedirs(os.path.dirname(path2save), exist_ok=True)
    with open(path2save, "w") as write_file:
        write_file.write(messages_JSON)