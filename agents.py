from groq import Groq

from config import GROQ_API_KEY, MODEL

client = Groq(api_key=GROQ_API_KEY)

CATEGORIES = ("HR", "TECHNICAL", "PROJECT", "GENERAL")

SPECIALIST_NAMES = {
    "HR": "HR specialist agent",
    "TECHNICAL": "Technical specialist agent",
    "PROJECT": "Project specialist agent",
    "GENERAL": "General company knowledge agent",
}

CATEGORY_TO_DEPARTMENT = {
    "HR": "hr",
    "TECHNICAL": "technical",
    "PROJECT": "projects",
    "GENERAL": None,
}

def _chat(messages, temperature=0.1, max_tokens=700):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()

def _keyword_fallback(question):
    q = question.lower()
    hr_terms = {
        "leave", "attendance", "salary", "payroll", "employee",
        "hr", "holiday", "remote work", "work from home", "benefit",
        "casual", "sick", "policy", "joining", "resignation",
    }
    technical_terms = {
        "python", "deploy", "deployment", "docker", "linux",
        "api", "server", "database", "code", "technical",
        "repository", "testing", "production", "framework",
    }
    project_terms = {
        "project", "alpha", "beta", "owner", "api owner",
        "architecture", "module", "team", "technology",
    }
    if any(term in q for term in project_terms):
        return "PROJECT"
    if any(term in q for term in hr_terms):
        return "HR"
    if any(term in q for term in technical_terms):
        return "TECHNICAL"
    return "GENERAL"

def manager_agent(question):
    prompt = f"""
Classify the employee's question into exactly one category:
HR, TECHNICAL, PROJECT, GENERAL.

HR = employee policies, leave, attendance, benefits, payroll, workplace policies.
TECHNICAL = programming, deployment, infrastructure, APIs, databases, engineering.
PROJECT = internal project ownership, project architecture, project technologies, project teams.
GENERAL = anything else.

Return ONLY the category word.

Question:
{question}
""".strip()

    try:
        result = _chat(
            [
                {"role": "system", "content": "You classify enterprise questions. Return only one category."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=10,
        ).upper()

        for category in CATEGORIES:
            if category in result:
                return category
    except Exception:
        pass

    return _keyword_fallback(question)

def generate_answer(question, context, category, history=None):
    if not context.strip():
        return "I could not find this information in the available company documents."

    specialist = SPECIALIST_NAMES.get(category, SPECIALIST_NAMES["GENERAL"])

    history_text = ""
    if history:
        history_text = "\n".join(
            f"{item['role'].upper()}: {item['content']}"
            for item in history[-6:]
        )

    prompt = f"""
You are the {specialist} in an enterprise knowledge assistant.

Answer the user's question using ONLY the supplied company-document context.
Do not use outside knowledge.
Do not invent names, policies, numbers, dates, systems, technologies, or ownership.

If the context does not contain enough information, reply exactly:
I could not find this information in the available company documents.

Keep the answer concise and professional.
When useful, mention the relevant source filename and page naturally.

Conversation context:
{history_text or "No previous conversation."}

Company-document context:
{context}

User question:
{question}
""".strip()

    try:
        return _chat(
            [
                {"role": "system", "content": "You are a grounded enterprise RAG answer agent."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=700,
        )
    except Exception as exc:
        return f"Unable to generate the answer right now: {exc}"
