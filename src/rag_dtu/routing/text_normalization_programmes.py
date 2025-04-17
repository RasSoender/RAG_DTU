import json
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
from config import get_openai_api_key


# Initialize NLP tools
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
client = OpenAI(api_key=get_openai_api_key())
# Constants
FIELDS_TO_SKIP = [
    "Curriculum, previous admission years",
    "Exam rules", # skip for now because it is too long for one embedding
]

def to_snake_case(s):
    """Convert a string to snake_case format."""
    s = s.strip().lower()
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', '_', s)
    return s


def clean_text(text):
    """Clean up spacing and punctuation in text."""
    text = re.sub(r'\s+([.,:%])', r'\1', text)  # remove space before punctuation
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text) # Remove markdown links, keeping only the text inside brackets
    text = text.replace("\n", " ").replace("*", "") # Remove special characters: asterisks (*) and newlines (\n)
    text = re.sub(r'\s+', ' ', text)  # normalize spaces
    return text.strip()

def extract_degree_name(text):
    # Define patterns to match
    patterns = [
        r"master of science in engineering\s*",  # Match "master of science in engineering" + space
        r"master of science\s*"                  # Match "master of science" + space
    ]
    
    for pattern in patterns:
        match = re.match(pattern, text, re.IGNORECASE)  # Check if the text starts with the pattern
        if match:
            return text[match.end():].strip()  # Extract everything after the matched phrase
    
    print(f"❌ No match found in the text: {text}")
    return text  # Return the original text if no match

def merge_curriculum_info(data):
    """
    Merge the values of 'programme_provision', 'specializations', and 'curriculum'
    into a single key 'curriculum_info' in the given dictionary.
    
    The function concatenates the values (if available) and uses a newline as a separator.
    After merging, the original keys are removed from the dictionary.
    
    Parameters:
        data (dict): The input dictionary containing the keys.
        
    Returns:
        dict: The updated dictionary with 'curriculum_info' and without the original keys.
    """
    keys_to_merge = ["programme_provision", "specializations", "curriculum"]
    merged_values = []
    
    # Collect values from the keys to merge
    for key in keys_to_merge:
        value = data.get(key, "").strip()
        if value:
            merged_values.append(value)
    
    # Create the merged string with newlines as delimiters
    merged_text = "\n".join(merged_values)
    
    # Assign the merged text to a new key 'curriculum_info'
    data["curriculum_info"] = merged_text
    
    # Remove the original keys
    for key in keys_to_merge:
        data.pop(key, None)
    
    return data
    
def normalize_programme_entry(entry):
    """Normalize programme entry structure."""

    normalized = {}

    # 1. Split programme title
    if "Official title" in entry:
        match = re.search(r"Master.*", entry["Official title"])

        # Extract and print the matched substring
        if match:
            result = match.group(0)
            print(f"Matched programme title: {result}")
            normalized["programme_name"] = result.strip()
        else:
            print("No match found in the programme title.")
            normalized["programme_name"] = entry["Official title"].strip()
    
    new_names = {
        "About the Programme Specification": "programme_specification",
        "Duration": "duration",
        "General admission requirements": "admission_requirements",
        "Academic requirements for this programme": "academic_requirements",
        "Competence profile": "competence_profile",
        "Programme specific goals for learning outcome": "learning_objectives",
        "Structure": "structure",
        "Programme provision": "programme_provision",
        "Specializations": "specializations",
        "Curriculum": "curriculum",
        "Curriculum, previous admission years": "curriculum_previous_years",
        "Master's thesis": "master_thesis",
        "Master thesis, specific rules": "master_thesis_specific_rules",
        "Study Activity Requirements and Deadlines": "activity_requirements",
        "Study Programme Rules": "programme_rules",
        "Regarding course descriptions": "course_descriptions",
        "Course registration": "course_registration",
        "Binding courses": "binding_courses",
        "Academic prerequisites for course participation": "academic_prerequisites",
        "Participation in limited admission courses": "limited_admission_courses",
        "Mandatory participation in class and exam prerequisites": "mandatory_participation",
        "Deadlines for publication of teaching material and syllabus": "teaching_material_deadlines",
        "Project courses": "project_courses",
        "Evaluation of teaching": "evaluation_of_teaching",
        "Exam rules": "exam_rules",
        "Credit Transfer, Studying Abroad, Exemption, Leave, etc.": "other_info",
        "Head of study": "head_of_study",
    }

    # 4. Copy remaining fields (unless skipped)
    for key, value in entry.items():
        if key in FIELDS_TO_SKIP or key in ["Official title"]:
            continue

        if isinstance(value, str):
            value = re.sub(r'\s+', ' ', value).strip()

        normalized[new_names[key]] = value


    return normalized

def normalize_text_fields(entry):
    """Normalize and clean text fields in a programme entry."""

    #entry name print

    normalized = {}

    for key, value in entry.items():
        new_key = to_snake_case(key)
        
        if isinstance(value, str):
            value = clean_text(value)

        normalized[new_key] = value

    return normalized

def summarization(text, model = "gpt-4o-mini-2024-07-18"):
    """Summarize text using OpenAI's GPT model."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": f"Summarize the following text including all informations and the summarization should be thought for a retrieval application: {text}"}
            ],
            max_tokens=500,
            temperature=0.5
        )
        summary = response.choices[0].message.content
        return summary
    except Exception as e:
        print(f"Error during summarization: {e}")
        return text  # Return original text if summarization fails

def preprocess_programme_text(programme_dict,programme_name, do_stemming=False, do_lemmatization=False):
    """Preprocess programme text for NLP analysis."""
    summarized_programme_dict = {}  
    for key, value in programme_dict.items():

        print(f"Processing field: {key}")
        # 3. Lowercase
        text = value.lower()

        # 4. Remove punctuation
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

        # 5. Tokenization
        tokens = word_tokenize(text)

        summ_text = ""
        # 6. Summarization

        if len(tokens) > 800:
            # 6. Summarization
            text_summarized = summarization(value)
            tex = text_summarized.lower()
            # Remove punctuation
            tex = re.sub(f"[{re.escape(string.punctuation)}]", " ", text_summarized)
            tokens_summarized = word_tokenize(tex)
            summ_text = " ".join(tokens_summarized)
            

        # 6. Remove stopwords
        #tokens = [word for word in tokens if word not in stop_words and word.isalpha()]

        # 7. Stemming / Lemmatization
        if do_stemming and not do_lemmatization:
            tokens = [stemmer.stem(word) for word in tokens]
        elif do_lemmatization and not do_stemming:
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
        elif do_stemming and do_lemmatization:
            # First lemmatization, then stemming
            tokens = [stemmer.stem(lemmatizer.lemmatize(word)) for word in tokens]

        # 8. Rebuild the text
        processed_text = " ".join(tokens)
        programme_dict[key] = processed_text
        summarized_programme_dict[key] = summ_text

    return programme_dict, summarized_programme_dict


def build_programme_data(programme_dict,name, do_stemming=False, do_lemmatization=False):
    """Build a processed programme data object from raw programme data."""
    # Step 1: Normalize structure and texts

    normalized = normalize_programme_entry(programme_dict)
    normalized = normalize_text_fields(normalized)
    new_dict = merge_curriculum_info(normalized)




    # print(f"Normalized programme data: {normalized}")
    # Step 2: Build text before preprocessing (combine string values)
    # preprocessed_text = " ".join(str(v) for v in normalized.values() if isinstance(v, str))

    # Step 3: Complete preprocessing
    postprocessed_dict, summarized_dict = preprocess_programme_text(normalized,name, do_stemming=do_stemming, do_lemmatization=do_lemmatization)

    programme_name = extract_degree_name(normalized.get("programme_name"))
    
    # Step 4: Add metadata
    postprocessed_dict["metadata"] = {
        "programme_name": programme_name
    }

    return postprocessed_dict, summarized_dict


def main():
    """Main function to execute the script on all programmes."""
    try:
        # Load JSON file with programme entries
        with open("data/all_programmes_info.json", "r", encoding="utf-8") as f:
            programmes = json.load(f)

        processed_programmes = {}

        summarized_programmes = {}

        print(f"Processing {len(programmes)} programmes...")
        
        # Process all programmes
        for programme_name, programme_data in programmes.items():
                
            processed, summ= build_programme_data(programme_data, name=programme_name, do_stemming=False, do_lemmatization=False)
            processed_programmes[programme_name] = processed
            summarized_programmes[programme_name] = summ

        
        # Save the processed output to a new JSON file
        output_file = "data/processed_programmes.json"
        summarized_output_file = "data/summarized_programmes.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_programmes, f, indent=2, ensure_ascii=False)
        with open(summarized_output_file, "w", encoding="utf-8") as f:
            json.dump(summarized_programmes, f, indent=2, ensure_ascii=False)

        print(f"✅ Processed {len(processed_programmes)} programmes and saved to '{output_file}'")
        print(f"✅ Summarized {len(summarized_programmes)} programmes and saved to '{summarized_output_file}'")
        
    except FileNotFoundError:
        print("❌ Error: Could not find input file 'all_programmes_info.json'")
    except json.JSONDecodeError:
        print("❌ Error: Invalid JSON format in input file")

def normalize_query(query, do_stemming=False, do_lemmatization=False, remove_stopwords=False):
    """
    Normalize and preprocess a search query for consistent text matching.
    
    Args:
        query (str): The input search query to normalize
        do_stemming (bool): Whether to apply Porter stemming
        do_lemmatization (bool): Whether to apply WordNet lemmatization
        remove_stopwords (bool): Whether to remove English stopwords
    
    Returns:
        tuple: (normalized query string, list of tokens)
    """
    # 1. Lowercase
    text = query.lower()

    # 2. Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

    # 3. Tokenization
    tokens = word_tokenize(text)

    # 4. Optional stopwords removal
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words and word.isalpha()]

    # 5. Stemming / Lemmatization
    if do_stemming and not do_lemmatization:
        tokens = [stemmer.stem(word) for word in tokens]
    elif do_lemmatization and not do_stemming:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    elif do_stemming and do_lemmatization:
        # First lemmatization, then stemming
        tokens = [stemmer.stem(lemmatizer.lemmatize(word)) for word in tokens]

    # 6. Rebuild the text
    normalized_query = " ".join(tokens)

    return normalized_query, tokens


if __name__ == "__main__":
    main()