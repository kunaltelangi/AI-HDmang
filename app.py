from flask import Flask, render_template, request, session
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, ChatMessagePromptTemplate

app = Flask(__name__)
app.secret_key = "secret-key"

template = """
Here is the conversation history:
{context}

Question: {question}

Please provide a detailed, step-by-step solution. Format each step on a new line without numbering. Keep explanations clear and concise.

Answer:
"""

model = OllamaLLM(model='dolphin-llama3', base_url='http://localhost:11434')
message_template = ChatMessagePromptTemplate.from_template(template, role="user")
prompt = ChatPromptTemplate.from_messages([message_template])
chain = prompt | model

@app.route("/", methods=["GET", "POST"])
def index():
    if "history" not in session:
        session["history"] = ""

    response = ""
    if request.method == "POST":
        question = request.form["question"]
        history = session["history"]

        response = chain.invoke({"context": history, "question": question})
        session["history"] += f"User: {question}\nAI: {response}\n"

    return render_template("index.html", response=response, history=session["history"])

if __name__ == "__main__":
    app.run(debug=True)
