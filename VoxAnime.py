from transformers import pipeline

# Load the NLP model
nlp = pipeline("text2text-generation", model="google/flan-t5-large", framework="pt")  # Try flan-t5-large for better results

def analyze_story(story):
    prompt = f"""
    Read the following story and extract important details.

    **Characters:** Who are the main characters? Provide names and any described appearances.
    **Scenes:** What are the settings? Describe the key actions happening in each place.
    **Emotions:** What emotions are present in the story? Describe how the characters feel.

    Do not include any extra text—just the details.

    Story: {story}
    """

    result = nlp(prompt, max_length=500)
    return result[0]['generated_text']

# Test it!
story = "A woman goes to the farmers market to buy groceries so she can prepare a meal for her family."
analysis = analyze_story(story)
print(analysis)
