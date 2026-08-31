"""Entry point for the GitHub digest agent.

Wires the hexagonal architecture adapters into the orchestrator and runs
the pipeline. Credentials are read from environment variables.

Usage:
    GITHUB_TOKEN=ghp_xxx GITHUB_REPOS=owner/repo OPENROUTER_API_KEY=sk-or-xxx \\
    GMAIL_ADDRESS=user@gmail.com GMAIL_APP_PASSWORD=xxx RECIPIENT_EMAIL=user@gmail.com \\
    python main.py
"""

import os

from dotenv import load_dotenv

from digest_agent.adapters.driven.github_adapter import GitHubAdapter
from digest_agent.adapters.driven.openrouter_adapter import OpenRouterAdapter
from digest_agent.adapters.driven.groq_adapter import GroqAdapter
from digest_agent.adapters.driven.mistral_adapter import MistralAdapter
from digest_agent.adapters.driven.fallback_provider import FallbackLLMProvider
from digest_agent.adapters.driven.reportlab_adapter import ReportLabAdapter
from digest_agent.adapters.driven.gmail_adapter import GmailAdapter
from digest_agent.core.orchestrator import build_orchestrator
from digest_agent.core.state import DigestState
from digest_agent.ports import LLMProviderPort


def build_llm_provider() -> LLMProviderPort:
    """Build a fallback LLM provider that tries Groq, then Mistral, then
    OpenRouter. Providers without a configured API key are skipped."""
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPEN_ROUTER_API_KEY", "")
    groq_key = os.environ.get("GROQ_API_KEY", "")
    mistral_key = os.environ.get("MISTRAL_API_KEY", "")

    providers = []
    if groq_key:
        providers.append(GroqAdapter(api_key=groq_key))
    if mistral_key:
        providers.append(MistralAdapter(api_key=mistral_key))
    if openrouter_api_key:
        providers.append(OpenRouterAdapter(api_key=openrouter_api_key))

    if not providers:
        raise ValueError(
            "No LLM provider configured. Set one of GROQ_API_KEY, "
            "MISTRAL_API_KEY, or OPENROUTER_API_KEY."
        )
    return FallbackLLMProvider(providers)


def main():
    # Load optional .env file (values are also read from real environment)
    load_dotenv()

    # Read configuration from environment variables
    github_token = os.environ.get("GITHUB_TOKEN", "")
    github_repos = os.environ.get("GITHUB_REPOS", "")
    lookback_days = int(os.environ.get("LOOKBACK_DAYS", "1"))
    gmail_address = os.environ.get("GMAIL_ADDRESS", "")
    gmail_app_password = os.environ.get("GMAIL_APP_PASSWORD", "")
    recipient_email = os.environ.get("RECIPIENT_EMAIL", "")

    llm_provider = build_llm_provider()

    if not all([github_token, github_repos]):
        print("Error: Missing required environment variables.")
        print("Required: GITHUB_TOKEN, GITHUB_REPOS")
        print("Optional: GROQ_API_KEY, MISTRAL_API_KEY, OPENROUTER_API_KEY, GMAIL_ADDRESS, RECIPIENT_EMAIL")
        return

    # Create the driven adapters
    github_adapter = GitHubAdapter(token=github_token, repos=github_repos, lookback_days=lookback_days)
    reportlab_adapter = ReportLabAdapter()
    gmail_adapter = GmailAdapter()

    # Build the orchestrator with the wired adapters
    orchestrator = build_orchestrator(
        activity_source=github_adapter,
        llm_provider=llm_provider,
        document_renderer=reportlab_adapter,
        notifier=gmail_adapter,
        max_attempts=5,
    )

    # Run the pipeline with a proper DigestState
    initial_state = DigestState(
        repos=[r.strip() for r in github_repos.split(",") if r.strip()],
        lookback_days=lookback_days,
        recipient=recipient_email or gmail_address,
    )
    result = orchestrator.invoke(initial_state)

    # Print summary
    print(f"Outcome: {result.outcome}")
    if result.outcome == "success":
        print(f"Summary: {result.summary[:100]}...")
        print(f"PDF: {result.pdf_path}")
        print(f"Email sent: {result.email_sent}")
    elif result.error_message:
        print(f"Error: {result.error_message}")


if __name__ == "__main__":
    main()