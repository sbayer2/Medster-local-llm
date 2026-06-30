import os

# Silence the benign "PyTorch was not found" advisory from HuggingFace
# transformers. mlx_vlm uses transformers only for the tokenizer/processor
# (chat template); we run on MLX, not PyTorch, so the missing-torch backend
# warning is irrelevant. Must be set before transformers is first imported.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from medster.agent import Agent
from medster.utils.intro import print_intro
from medster import config
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory


def main():
    print_intro()

    # Model selection prompt
    print("\n" + "="*70)
    print("MODEL SELECTION")
    print("="*70)
    backend = ("OptiQ 4-bit via mlx_vlm — single model, on-device, no Ollama"
               if config.OPTI_ALL_MODE
               else "Ollama (agent loop) + OptiQ via mlx_vlm (vision)")
    print(f"\nActive backend: {backend}")
    print(f"  (OPTI_ALL_MODE={config.OPTI_ALL_MODE} — change in .env)")
    print("\nChoose your model:")
    print("\n1. qwen3.6:35b-mlx (PRIMARY - TEXT + VISION)")
    if config.OPTI_ALL_MODE:
        print("   - With OPTI_ALL_MODE on, runs as OptiQ 4-bit (Qwen3.6-35B-A3B)")
        print("     via mlx_vlm on-device — NOT Ollama's text MLX model")
    print("   - Qwen3.6 35B-A3B Mixture-of-Experts with Vision")
    print("   - 128K context window, ~3.5B active params (fast on Apple Silicon)")
    print("   - Clinical reasoning, labs, notes, reports")
    print("   - Vision: DICOM images, ECG tracings, X-rays, medical documents")
    print("   - High tool call reliability (0.92)")
    print("\n2. gpt-oss:20b (DEPRECATED - TEXT-ONLY)")
    print("   - Legacy model - kept for backwards compatibility")
    print("   - Faster inference, text-only")
    print("\n3. qwen3-vl:8b (DEPRECATED - TEXT + VISION)")
    print("   - Legacy model - kept for backwards compatibility")
    print("   - Slower inference, smaller vision model")
    print("\n" + "="*70)

    while True:
        choice = input("\nEnter your choice (1, 2, or 3): ").strip()
        if choice == "1":
            model_name = "qwen3.6:35b-mlx"
            _be = "OptiQ via mlx_vlm" if config.OPTI_ALL_MODE else "Ollama"
            print(f"\n✓ Selected: qwen3.6:35b-mlx (text + vision) — {_be}")
            break
        elif choice == "2":
            model_name = "gpt-oss:20b"
            print(f"\n✓ Selected: gpt-oss:20b (text-only, deprecated)")
            break
        elif choice == "3":
            model_name = "qwen3-vl:8b"
            print(f"\n✓ Selected: qwen3-vl:8b (vision, deprecated)")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

    print("="*70 + "\n")

    # Set the selected model globally so tools can use it
    config.set_selected_model(model_name)

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
