import datetime
import os
import json


# Function to extract the ID from the image path
def extract_id(image_path):
    return os.path.basename(image_path).split('.')[0]


# Function to convert a single conversation JSON into the required format
def convert_conversation(conversation_json, image_folder):
    conversations = []
    image_path = ""

    for conv in conversation_json["conversations"]:
        if "value" in conv and "/Users/ansingh/Desktop/hackathon/project/" in conv["value"]:
            # Extract image file name and set the new image path using image_folder
            image_path = conv["value"]
            image_name = os.path.basename(image_path)
            new_image_path = os.path.join(image_folder, image_name)

            conversations.append({
                "from": "user",
                "value": f"Picture 1: <img>{new_image_path}</img>"
            })
        else:
            conversations.append({
                "from": conv["role"],
                "value": conv["content"]
            })

    # Extract the ID from the image path
    conv_id = extract_id(image_path)

    return {
        "id": f"identity_{conv_id}",
        "conversations": conversations
    }


# Function to combine JSON files from a given directory
def combine_json_files_from_directory(directory, image_folder):
    combined_conversations = []

    # List all JSON files in the directory
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]

    for json_file in json_files:
        with open(os.path.join(directory, json_file), 'r') as f:
            conversation_json = json.load(f)
            combined_conversations.append(convert_conversation(conversation_json, image_folder))

    return combined_conversations


# Path to the directory containing JSON files
directory = '/Users/daneja/Downloads/project_2/'

# Path to the folder where images will be stored
image_folder = '/Users/daneja/Downloads/project_2/'  # Set your new image folder path here

# Combine the JSON files from the directory
combined_result = combine_json_files_from_directory(directory, image_folder)

# Save the combined result to a new file
output_file = f'combined_conversations_{datetime.date.today()}.json'
with open(output_file, 'w') as f:
    json.dump(combined_result, f, indent=4)

print(f"Combined conversations saved successfully at {output_file}!")
