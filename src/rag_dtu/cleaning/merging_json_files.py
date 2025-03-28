import os
import json

# Folder containing all the course JSON files
folder_path = 'data/data_courses'

# Initialize an empty dictionary to store all course data
all_courses = {}

# Loop through each file in the folder
i = 1
for filename in os.listdir(folder_path):
    if i % 50 == 0:
        print(i)
    i += 1
    if filename.endswith('.json'):
        # Construct full file path
        file_path = os.path.join(folder_path, filename)

        # Read the JSON content from the file
        with open(file_path, 'r', encoding='utf-8') as file:
            course_data = json.load(file)

        # Extract the course code (assuming it's in the course title)
        course_code = filename.split('.')[0]  # Get the file name without extension

        # Add the course data to the all_courses dictionary
        all_courses[course_code] = course_data

# Save the combined data into a new JSON file
output_file = 'data/all_courses_info.json'
with open(output_file, 'w', encoding='utf-8') as output:
    json.dump(all_courses, output, ensure_ascii=False, indent=4)

print(f"Combined data has been saved to {output_file}")