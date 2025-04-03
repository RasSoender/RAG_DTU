import os
import json

titles = [
    "Official title",
    "About the Programme Specification",
    "Duration",
    "General admission requirements",
    "Academic requirements for this programme",
    "Competence profile",
    "Programme specific goals for learning outcome",
    "Structure",
    "Programme provision",
    "Specializations",
    "Curriculum",
    "Curriculum, previous admission years",
    "Master's thesis",
    "Master thesis, specific rules",
    "Study Activity Requirements and Deadlines",
    "Study Programme Rules",
    "Regarding course descriptions",
    "Course registration",
    "Binding courses",
    "Academic prerequisites for course participation",
    "Participation in limited admission courses",
    "Mandatory participation in class and exam prerequisites",
    "Deadlines for publication of teaching material and syllabus",
    "Project courses",
    "Evaluation of teaching",
    "Exam rules",
    "Credit Transfer, Studying Abroad, Exemption, Leave, etc.",
    "Head of study"
]   


def clean_data(text):
    scraped_info = {}

    lines = text.split("\n")  # Split text into lines
    
    title_n = len(titles)

    title_index = 1
    current_title = titles[0]
    next_title = titles[1] if title_index < title_n else None
    current_content = []

    for line in lines:
        line = line.strip()

        if line.startswith("## "):
            line = line[3:].strip()  # Remove "## " and strip the rest
            if line == "Official title":
                current_content = []

        if next_title and line == next_title:
            scraped_info[current_title] = "\n".join(current_content).strip()
            
            current_title = next_title
            current_content = []
            title_index += 1
            next_title = titles[title_index] if title_index < title_n else None
        else:
            current_content.append(line)

    # Save the last section
    if current_title:
        if current_title == "Head of study":
            scraped_info[current_title] = "\n".join(current_content[:5]).strip()
        else: 
            scraped_info[current_title] = "\n".join(current_content).strip()

    return scraped_info


# Following function should only be used for the program called 'Technology Entrepreneurs'
def cleaning_specific_program(text):
    titles = [
    "Official title",
    "About the programme, Cand tech.",
    "Duration, Cand tech.",
    "Academic requirements for this programme",
    "Competence profile, Cand. tech.",
    "Programme specific goals for learning outcome",
    "Structure, Cand tech.",
    "Programme provision",
    "Specializations",
    "Curriculum",
    "Curriculum, previous admission years",
    "Master's thesis",
    "Master thesis, specific rules",
    "Study Activity Requirements and Deadlines",
    "Study Programme Rules",
    "Regarding course descriptions",
    "Course registration",
    "Binding courses",
    "Academic prerequisites for course participation",
    "Participation in limited admission courses",
    "Mandatory participation in class and exam prerequisites",
    "Deadlines for publication of teaching material and syllabus",
    "Project courses",
    "Evaluation of teaching",
    "Exam rules",
    "Credit Transfer, Studying Abroad, Exemption, Leave, etc.",
    "Head of study"
    ]   

    scraped_info = {}

    lines = text.split("\n")  # Split text into lines
    
    title_n = len(titles)

    title_index = 1
    current_title = titles[0]
    next_title = titles[1] if title_index < title_n else None
    current_content = []

    for line in lines:
        line = line.strip()

        if line.startswith("## "):
            line = line[3:].strip()  # Remove "## " and strip the rest
            if line == "Official title":
                current_content = []

        if next_title and line == next_title:
            if current_title.endswith(", Cand. tech."):
                current_title = current_title[:-len(", Cand. tech.")]
            elif current_title.endswith(", Cand tech."):
                current_title = current_title[:-len(", Cand. tech.")]
            
            scraped_info[current_title] = "\n".join(current_content).strip()
            
            current_title = next_title
            current_content = []
            title_index += 1
            next_title = titles[title_index] if title_index < title_n else None
        else:
            current_content.append(line)

    # Save the last section
    if current_title:
        if current_title == "Head of study":
            scraped_info[current_title] = "\n".join(current_content[:5]).strip()
        else: 
            scraped_info[current_title] = "\n".join(current_content).strip()

    scraped_info["General admission requirements"] = ""

    return scraped_info




# Folder containing all the course JSON files
folder_path = 'data/data_study_programmes'

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):

        # Construct full file path
        file_path = os.path.join(folder_path, filename)

        # Read the JSON content from the file
        with open(file_path, 'r', encoding='utf-8') as file:
            course_data = json.load(file)

        programme_info = clean_data(course_data["markdown"])

        # Save the combined data into a new JSON file
        output_file = f'data/data_study_programmes_cleaned/{filename}'

        with open(output_file, 'w', encoding='utf-8') as output:
             json.dump(programme_info, output, ensure_ascii=False, indent=4)


print(f"The markdown files of the study programmes has been saved into {output_file}")