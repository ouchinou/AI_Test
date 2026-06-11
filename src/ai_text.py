import ollama


def generer_texte(prompt):
    response = ollama.chat(
        model="llama3.1:8b", messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


if __name__ == "__main__":
    print(generer_texte("Donne-moi une astuce de code Python."))
