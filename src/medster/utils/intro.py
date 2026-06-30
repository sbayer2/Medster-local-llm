from medster.utils.ui import Colors


def print_intro():
    """Print the Medster ASCII art and introduction."""

    ascii_art = f"""
{Colors.CYAN}{Colors.BOLD}
 ███╗   ███╗███████╗██████╗ ███████╗████████╗███████╗██████╗
 ████╗ ████║██╔════╝██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔══██╗
 ██╔████╔██║█████╗  ██║  ██║███████╗   ██║   █████╗  ██████╔╝
 ██║╚██╔╝██║██╔══╝  ██║  ██║╚════██║   ██║   ██╔══╝  ██╔══██╗
 ██║ ╚═╝ ██║███████╗██████╔╝███████║   ██║   ███████╗██║  ██║
 ╚═╝     ╚═╝╚══════╝╚═════╝ ╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
{Colors.ENDC}
"""

    subtitle = f"{Colors.DIM}Autonomous Clinical Case Analysis Agent{Colors.ENDC}"


    info = f"""
{Colors.CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.ENDC}
{Colors.DIM}  Powered by: Qwen3.6-35B-A3B OptiQ (MLX, 100% on-device) + Coherent FHIR/DICOM{Colors.ENDC}
{Colors.DIM}  Primary Use Case: Clinical Case Analysis{Colors.ENDC}
{Colors.DIM}  Type 'exit' or 'quit' to end session{Colors.ENDC}
{Colors.CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.ENDC}

{Colors.YELLOW}⚠  DISCLAIMER: For research and educational purposes only.{Colors.ENDC}
{Colors.YELLOW}   Not for clinical decision-making without physician review.{Colors.ENDC}
"""

    print(ascii_art)
    print(subtitle)
    print(info)
