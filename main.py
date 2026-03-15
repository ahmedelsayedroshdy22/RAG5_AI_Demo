from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory="cdr_vectorstore",
    embedding_function= embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

model = OllamaLLM(model="llama3.2")

template = """
You are a Voice BOT Assistant developed by Ahmed Zayed 2026 .

STRICT RULE: Every single response MUST begin with this exact sentence, unchanged:
"I am a Voice BOT Assistant developed by Ahmed Zayed 2026"
Never skip this. Never rephrase it.

You help NOC/voice engineers diagnose and resolve call issues based on CDR data.
Use ONLY the context below to answer. If the answer is not in the context, say:
"I don't have enough information in my knowledge base for this."

Relevant knowledge base context:
{context}

Engineer's question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    user_question = input("\nAsk me about the CDR (or type 'q' to exit): ").strip()

    if user_question.lower() == 'q':
        print("Goodbye!")
        break

    if not user_question:
        print("Please type a question.")
        continue

    relevant_docs = retriever.invoke(user_question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    result = chain.invoke({
        "context": context,
        "question": user_question
    })

    print(f"\nBot: {result}\n")
