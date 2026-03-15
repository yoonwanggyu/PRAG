### Document Rewriting
# Details of the prompts used to rewrite or transform documents for augmentation purposes.

rewrite_prompt = """
Rewrite the following passage. While keeping the entities, proper nouns, and key details such as names, locations, and terminology intact, create a new version of the text that expresses the same ideas in a different way. Make sure the revised passage is distinct from the original one, but preserves the core meaning and relevant information.
{passage}
"""

### QA Generation
# Explanation of the prompts used to generate question-answer pairs for document augmentation.

qa_prompt = """
I will provide a passage of text, and you need to generate three different questions based on the content of this passage. Each question should be answerable using the information provided in the passage. Additionally, please provide an appropriate answer for each question derived from the passage.
You need to generate the question and answer in the following format:
[
    {
        "question": "What is the capital of France?",
        "answer": "Paris"
        "full_answer": "The capital of France is Paris."
    }, 
]
This list should have at least three elements. You only need to output this list in the above format.
Passage:
{passage}
"""