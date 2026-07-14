import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smart_docs.agent import SmartDocumentAgent
from smart_docs.rag import RAGService


EVAL_SET = [
    {
        "question": "How many weeks of paid parental leave do primary caregivers get?",
        "keywords": ["16", "weeks", "parental"],
    },
    {
        "question": "What is the Q4 2025 operating budget?",
        "keywords": ["2,000,000", "operating", "budget"],
    },
    {
        "question": "What is 15% of the operating budget?",
        "keywords": ["300000", "300,000"],
    },
    {
        "question": "What MFA methods are approved?",
        "keywords": ["authenticator", "hardware", "security"],
    },
    {
        "question": "What is the company pet policy?",
        "keywords": ["don't know", "provided documents"],
    },
]


def main() -> None:
    rag = RAGService()
    rag.ingest_existing_samples()
    agent = SmartDocumentAgent(rag)

    passed = 0
    for index, item in enumerate(EVAL_SET, start=1):
        response = agent.chat(f"eval-{index}", item["question"])
        answer = response.answer.lower().replace("$", "").replace(",", "")
        ok = any(keyword.lower().replace(",", "") in answer for keyword in item["keywords"])
        passed += int(ok)
        print(f"{index}. {'PASS' if ok else 'FAIL'} - {item['question']}")
        print(f"   {response.answer[:300]}")

    print(f"\nScore: {passed}/{len(EVAL_SET)}")


if __name__ == "__main__":
    main()
