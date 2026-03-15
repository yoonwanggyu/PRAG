from openai import OpenAI
import json,os

api_key = os.getenv("OPEN_AI_API")
client = OpenAI(api_key = api_key)

with open("data_generate_prompt.txt","r",encoding="utf-8") as f:
    instructions = f.read()

passage = """
El extraño viaje El extraño viaje is a 1964 Spanish black drama film directed by Fernando Fernán Gómez. Famous film director Jess Franco acts as the brother of the protagonist. The film was a huge flop on its limited release. It was voted seventh best Spanish film by professionals and critics in 1996 Spanish cinema centenary. In a large house in the middle of a little Spanish town live Venancio and Paquita, the retarded brother and sister of Ignacia, who bullies them continuously. Suspecting that she has a visitor after dark, they start snooping and one night she turns on.
used to slip into Ignacia's house after work. When he found her dead, he helped Venancio and Paquita dispose of the body. Then he took them away to the sea, where he gave them knockout drops so that he could escape with Ignacia's money. Unfortunately, his dose was too powerful. El extraño viaje El extraño viaje is a 1964 Spanish black drama film directed by Fernando Fernán Gómez. Famous film director Jess Franco acts as the brother of the protagonist. The film was a huge flop on its limited release. It was voted seventh best Spanish film by professionals and.
El Viaje de Rose El Viaje de Rose (it could be translated as “The Rose’s Journey”) are a Spanish pop-rock band from Badajoz, Extremadura. The band is led by Ana Broncano, the singer. The name El Viaje de Rose is due to the fact that Ana saw a poster of the film Titanic when they were thinking about the name of the band The band was created in 2004 and Broncano's lyrics have frequent allusions to love and sotial problems. Well-known songs by El Viaje de Rose include "Deseo", "No se puede matar al amor", "Tras la vida", "Caos", "Ayúdame"
"""

response = client.responses.create(
    model="gpt-5.4",
    # reasoning={"effort": "medium"},
    instructions=instructions,
    input=[
        {
            "role": "user",
            "content": f"Generate 100 questions based on the given passage following the instructions. Passage: \n\n {passage}"}
    ],
    temperature=0.7,
    max_output_tokens=6000
)

output = response.output_text
print(output)

# questions = json.loads(output)

# with open("generated_questions.json", "w", encoding="utf-8") as f:
#     json.dump(questions, f, ensure_ascii=False, indent=4)