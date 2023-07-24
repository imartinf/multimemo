import json
import os

import click


def save_json(json_data, path2save, json_indent):
    messages_JSON = json.dumps(json_data, indent=json_indent)
    # Create folder if it doesn't exist
    os.makedirs(os.path.dirname(path2save), exist_ok=True)
    with open(path2save, "w") as write_file:
        write_file.write(messages_JSON)

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        click.echo(f"Directory {dir_path} created.")
    else:
        click.echo(f"Directory {dir_path} already exists.")