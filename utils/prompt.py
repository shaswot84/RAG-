def build_prompt(context_chunks, query):
    context  = " ".join(context_chunks)
    return f"""Use the following context to answer the question.
    Context:
    {context}

    Question:
    {query}"""