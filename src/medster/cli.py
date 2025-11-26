from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from medster.agent import Agent
from medster.utils.intro import print_intro
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory


def main():
    print_intro()

    # Model selection prompt
    print("\n" + "="*70)
    print("MODEL SELECTION")
    print("="*70)
    print("\nChoose your model:")
    print("\n1. gpt-oss:20b (TEXT-ONLY)")
    print("   - Faster inference")
    print("   - Clinical reasoning, labs, notes, reports")
    print("   - Cannot process medical images")
    print("\n2. qwen3-vl:8b (TEXT + IMAGES)")
    print("   - Multimodal vision support")
    print("   - Can analyze DICOM images, ECG tracings, X-rays")
    print("   - Slower inference")
    print("\n" + "="*70)

    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()
        if choice == "1":
            model_name = "gpt-oss:20b"
            print(f"\n✓ Selected: gpt-oss:20b (text-only)")
            break
        elif choice == "2":
            model_name = "qwen3-vl:8b"
            print(f"\n✓ Selected: qwen3-vl:8b (text + images)")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    print("="*70 + "\n")

    agent = Agent(model_name=model_name)

    # Create a prompt session with history
    session = PromptSession(history=InMemoryHistory())

    while True:
        try:
            # Prompt the user for input
            query = session.prompt("medster>> ")
            if query.lower() in ["exit", "quit"]:
                print("Session ended. Goodbye!")
                break
            if query:
                # Run the clinical analysis agent
                agent.run(query)
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended. Goodbye!")
            break


if __name__ == "__main__":
    main()
