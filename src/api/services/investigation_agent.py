"""LangChain investigation agent with local TF-IDF RAG over curated fraud guidance."""
from functools import lru_cache
import os
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import select
from src.api.db import SessionLocal
from src.api.models import Transaction

KNOWLEDGE = [
    ("Account takeover", "Account takeover indicators include a new or unfamiliar device, abrupt changes in transaction amount, unusual access patterns, and a sender's activity deviating from their historical baseline. Verify device history and authenticate the account holder through established procedures."),
    ("Unusual large transfer", "A transfer materially above a customer's normal amount can warrant review. Compare amount with historical behavior, account balance and known customer context. A large amount alone is not proof of fraud."),
    ("Rapid repeated transfers", "Multiple transfers in a short interval, especially from one sender to multiple receivers or in repeated similar amounts, can indicate structuring or account compromise. Review timestamps, counterparties and the broader transaction sequence."),
    ("Cross-border activity", "Unexpected cross-border transfers can merit review when inconsistent with established customer behavior. Confirm geography, customer travel or business context and counterparty legitimacy before escalation."),
    ("Investigation workflow", "Treat model flags as triage signals, not conclusions. Corroborate with transaction history, account context and independent evidence. Record evidence, uncertainty and recommended human-review steps. Do not freeze, reject or block payments based on this demo."),
    ("Model limitations", "FinGuard AI is trained on synthetic data. Its risk score is not a calibrated real-world probability and must not be used for real financial decisions. Feature importance is global model metadata, not a per-transaction causal explanation.")
]
_VECTOR = TfidfVectorizer(stop_words="english")
_MATRIX = _VECTOR.fit_transform([f"{title}. {body}" for title, body in KNOWLEDGE])

def retrieve_knowledge(query: str, k: int = 3) -> str:
    scores = cosine_similarity(_VECTOR.transform([query]), _MATRIX)[0]
    indices = scores.argsort()[::-1][:k]
    return "\n\n".join(
        f"[{KNOWLEDGE[i][0]}] {KNOWLEDGE[i][1]}" for i in indices if scores[i] > 0
    ) or "No directly matching guidance was retrieved. State that the knowledge base did not provide supporting guidance."

@lru_cache
def _build_agent(api_key: str, model_name: str):
    llm = ChatGroq(api_key=api_key, model=model_name, temperature=0.1)

    @tool
    def search_fraud_guidance(query: str) -> str:
        """Retrieve relevant fraud patterns and investigation guidance from the FinGuard knowledge base."""
        return retrieve_knowledge(query)

    @tool
    def lookup_transaction(transaction_id: int) -> str:
        """Retrieve a saved FinGuard assessment by its numeric database transaction ID."""
        with SessionLocal() as db:
            item = db.get(Transaction, transaction_id)
            if item is None:
                return f"No saved transaction exists with database ID {transaction_id}."
            return (
                f"Database ID: {item.id}; reference: {item.external_id}; amount: {item.currency} {item.amount}; "
                f"channel: {item.channel}; sender: {item.sender_account}; receiver: {item.receiver_account}; "
                f"risk score: {item.risk_score}; risk level: {item.risk_level}; flagged: {item.is_flagged}; "
                f"model probability: {item.model_probability}; recommendation: {item.recommendation}; "
                f"model: {item.model_name}; created: {item.created_at}; feature values: {item.features}"
            )

    return create_agent(
        model=llm,
        tools=[search_fraud_guidance, lookup_transaction],
        system_prompt=(
            "You are FinGuard AI's fraud investigation assistant. You are an analyst support tool, not an autonomous decision-maker. "
            "Use search_fraud_guidance for relevant fraud patterns. If the user supplies a numeric saved transaction database ID, "
            "use lookup_transaction before discussing that record. Ground factual claims in retrieved tool results. "
            "If evidence is missing, say so; never invent transaction details, claim a fraud is proven, or claim a payment was blocked. "
            "Separate observed facts, model output, possible interpretation, uncertainty, and suggested human-review steps. "
            "The underlying model uses synthetic training data; scores are not calibrated real-world probabilities. "
            "Never request credentials, card numbers, passwords, OTPs or other sensitive authentication data."
        )
    )

def investigate(question: str):
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("The AI investigation assistant is not configured. Add GROQ_API_KEY to the Render service environment.")
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    agent = _build_agent(api_key, model_name)
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result.get("messages", [])
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content
    return "The investigation agent returned no text response."
