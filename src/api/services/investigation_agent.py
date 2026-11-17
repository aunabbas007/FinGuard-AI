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
    def lookup_recent_transactions(limit: int = 5) -> str:
        """Retrieve the most recent saved transaction assessments. Use this whenever the user asks about the latest, recent, or current transaction or asks a general question about recent activity."""
        safe_limit = max(1, min(int(limit), 10))
        with SessionLocal() as db:
            items = db.scalars(
                select(Transaction).order_by(Transaction.created_at.desc(), Transaction.id.desc()).limit(safe_limit)
            ).all()
            if not items:
                return "There are no recent transaction assessments available to review."
            return "\\n\\n".join(
                f"Transaction reference: {item.external_id or 'Not provided'}; assessment ID: {item.id}; "
                f"amount: {item.currency} {item.amount}; channel: {item.channel}; "
                f"risk score: {item.risk_score}; risk level: {item.risk_level}; "
                f"flagged for review: {item.is_flagged}; model probability: {item.model_probability}; "
                f"recommendation: {item.recommendation}; assessed at: {item.created_at}"
                for item in items
            )

    @tool
    def lookup_transaction(identifier: str) -> str:
        """Find a saved assessment by its external transaction reference (such as TXN006716) or numeric assessment ID."""
        value = identifier.strip()
        with SessionLocal() as db:
            item = None
            if value.isdigit():
                item = db.get(Transaction, int(value))
            if item is None:
                item = db.scalar(
                    select(Transaction).where(Transaction.external_id == value).order_by(Transaction.created_at.desc())
                )
            if item is None:
                return (
                    f"No assessment matching '{value}' was found in the recent transaction records. "
                    "Do not mention databases, SQLite, persistence, or internal storage to the user. "
                    "Offer to review the latest assessed transaction or ask for the exact reference."
                )
            return (
                f"Transaction reference: {item.external_id or 'Not provided'}; assessment ID: {item.id}; "
                f"amount: {item.currency} {item.amount}; channel: {item.channel}; "
                f"risk score: {item.risk_score}; risk level: {item.risk_level}; "
                f"flagged for review: {item.is_flagged}; model probability: {item.model_probability}; "
                f"recommendation: {item.recommendation}; model: {item.model_name}; "
                f"assessed at: {item.created_at}; feature values: {item.features}"
            )

    return create_agent(
        model=llm,
        tools=[search_fraud_guidance, lookup_recent_transactions, lookup_transaction],
        system_prompt=(
            "You are FinGuard AI's fraud investigation assistant. Speak directly to the user in clear, concise language; "
            "do not expose internal implementation details such as SQLite, database tables, storage, persistence, or tool names. "
            "For questions about the latest, recent, current, or general transaction activity, ALWAYS call lookup_recent_transactions first. "
            "For a supplied transaction reference such as TXN006716, call lookup_transaction; it supports both external references and numeric assessment IDs. "
            "Use search_fraud_guidance when relevant fraud-pattern context is needed. Ground transaction-specific claims only in returned transaction records. "
            "If a reference cannot be matched, do not say the user failed to save it or reveal internal storage details; explain simply that the reference could not be matched "
            "and offer to assess the latest available transaction or ask for the exact reference. "
            "When a recent transaction is available, answer the question directly using its amount, risk score, risk level, flagged status and recommendation. "
            "Separate observed facts from possible interpretations. Never invent facts, claim fraud is proven, or claim a payment was blocked. "
            "The underlying model uses synthetic training data; scores are not calibrated real-world probabilities. "
            "Never request credentials, full card numbers, passwords, OTPs or other sensitive authentication data."
        )
    )

def investigate(question: str):
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("The AI investigation assistant is not configured. Add GROQ_API_KEY to the Render service environment.")
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    agent = _build_agent(api_key, model_name)
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result.get("messages", [])
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content
    return "The investigation agent returned no text response."
