"""
Interactive Chat Interface for AutoStream Agent
User-friendly terminal chat with real-time responses
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Import agent
from langgraph_agent import AutoStreamAgent

# Load environment
load_dotenv()

# Colors for terminal (ANSI codes)
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header():
    """Print welcome header"""
    print("\n" + "="*70)
    print(f"{Colors.BOLD}{Colors.CYAN}🤖 AUTOSTREAM CONVERSATIONAL AI AGENT{Colors.ENDC}")
    print("="*70)
    print(f"{Colors.GREEN}✅ Agent initialized and ready!{Colors.ENDC}\n")


def print_instructions():
    """Print usage instructions"""
    print(f"{Colors.BOLD}Instructions:{Colors.ENDC}")
    print(f"  • Type your message and press Enter")
    print(f"  • Type {Colors.YELLOW}'exit'{Colors.ENDC} or {Colors.YELLOW}'quit'{Colors.ENDC} to end conversation")
    print(f"  • Type {Colors.YELLOW}'help'{Colors.ENDC} for available commands")
    print(f"  • Type {Colors.YELLOW}'state'{Colors.ENDC} to see full agent state")
    print(f"  • Type {Colors.YELLOW}'history'{Colors.ENDC} to see conversation history\n")


def print_state(agent: AutoStreamAgent):
    """Display current agent state"""
    state = agent.get_state()
    
    print(f"\n{Colors.BOLD}[Agent State]{Colors.ENDC}")
    print(f"  Turn: {state['turn_count']}")
    print(f"  Intent: {Colors.YELLOW}{state['intent']}{Colors.ENDC}")
    print(f"  Collecting: {Colors.YELLOW}{state['current_field_collecting']}{Colors.ENDC}")
    print(f"  Collected Info: {state['collected_info']}")
    print(f"  Lead Captured: {Colors.GREEN if state['lead_captured'] else Colors.RED}{state['lead_captured']}{Colors.ENDC}")
    print()


def print_conversation_history(agent: AutoStreamAgent):
    """Display full conversation history"""
    state = agent.get_state()
    messages = state['messages']
    
    if not messages:
        print(f"{Colors.YELLOW}No messages yet.{Colors.ENDC}\n")
        return
    
    print(f"\n{Colors.BOLD}[Conversation History ({len(messages)} messages)]{Colors.ENDC}\n")
    
    for i, msg in enumerate(messages, 1):
        role = msg['role']
        content = msg['content']
        
        if role == "user":
            print(f"{Colors.BLUE}👤 Turn {i} - You:{Colors.ENDC}")
            print(f"   {content}\n")
        else:
            print(f"{Colors.GREEN}🤖 Turn {i} - Agent:{Colors.ENDC}")
            print(f"   {content}\n")


def print_help():
    """Print help menu"""
    print(f"\n{Colors.BOLD}Available Commands:{Colors.ENDC}")
    print(f"  {Colors.YELLOW}exit{Colors.ENDC}      - End conversation")
    print(f"  {Colors.YELLOW}quit{Colors.ENDC}      - End conversation")
    print(f"  {Colors.YELLOW}help{Colors.ENDC}      - Show this help menu")
    print(f"  {Colors.YELLOW}state{Colors.ENDC}     - Show current agent state")
    print(f"  {Colors.YELLOW}history{Colors.ENDC}   - Show full conversation history")
    print(f"  {Colors.YELLOW}clear{Colors.ENDC}     - Clear screen")
    print()


def save_conversation(agent: AutoStreamAgent):
    """Save conversation history to file"""
    state = agent.get_state()
    messages = state['messages']
    
    if not messages:
        print(f"{Colors.YELLOW}No conversation to save.{Colors.ENDC}\n")
        return
    
    # Create conversations directory if it doesn't exist
    conv_dir = Path("conversations")
    conv_dir.mkdir(exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = conv_dir / f"conversation_{timestamp}.txt"
    
    # Write conversation to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write(f"AutoStream Agent Conversation\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        # Write messages
        for msg in messages:
            role = msg['role'].upper()
            content = msg['content']
            f.write(f"{role}:\n{content}\n\n")
        
        # Write final state
        f.write("="*70 + "\n")
        f.write("Final Agent State:\n")
        f.write(f"  Turns: {state['turn_count']}\n")
        f.write(f"  Intent: {state['intent']}\n")
        f.write(f"  Collected Info: {state['collected_info']}\n")
        f.write(f"  Lead Captured: {state['lead_captured']}\n")
        f.write("="*70 + "\n")
    
    print(f"{Colors.GREEN}✅ Conversation saved to: {filename}{Colors.ENDC}\n")


def print_summary(agent: AutoStreamAgent):
    """Print conversation summary"""
    state = agent.get_state()
    
    print(f"\n{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}CONVERSATION SUMMARY{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*70}{Colors.ENDC}\n")
    
    print(f"{Colors.BOLD}Statistics:{Colors.ENDC}")
    print(f"  Total Turns: {state['turn_count']}")
    print(f"  Final Intent: {Colors.YELLOW}{state['intent']}{Colors.ENDC}")
    print(f"  Lead Captured: {Colors.GREEN if state['lead_captured'] else Colors.RED}{state['lead_captured']}{Colors.ENDC}")
    
    if state['collected_info']:
        print(f"\n{Colors.BOLD}Collected Information:{Colors.ENDC}")
        for key, value in state['collected_info'].items():
            print(f"  • {key.capitalize()}: {Colors.YELLOW}{value}{Colors.ENDC}")
    
    print(f"\n{Colors.BOLD}Message Count:{Colors.ENDC}")
    user_msgs = sum(1 for m in state['messages'] if m['role'] == 'user')
    agent_msgs = sum(1 for m in state['messages'] if m['role'] == 'assistant')
    print(f"  User Messages: {user_msgs}")
    print(f"  Agent Messages: {agent_msgs}")
    
    print(f"\n{Colors.BOLD}{'='*70}{Colors.ENDC}\n")


def chat_loop():
    """Main interactive chat loop"""
    
    # Initialize agent
    try:
        agent = AutoStreamAgent()
    except Exception as e:
        print(f"{Colors.RED}❌ Error initializing agent: {e}{Colors.ENDC}")
        return
    
    # Print welcome
    print_header()
    print_instructions()
    
    # Main loop
    try:
        while True:
            # Get user input
            try:
                user_input = input(f"{Colors.BLUE}👤 You: {Colors.ENDC}").strip()
            except EOFError:
                # Handle Ctrl+D
                print(f"\n{Colors.YELLOW}Input ended.{Colors.ENDC}")
                break
            except KeyboardInterrupt:
                # Handle Ctrl+C
                print(f"\n{Colors.YELLOW}Interrupted by user.{Colors.ENDC}")
                break
            
            # Empty input
            if not user_input:
                print(f"{Colors.YELLOW}Please enter a message.{Colors.ENDC}\n")
                continue
            
            # Handle commands
            if user_input.lower() in ['exit', 'quit']:
                print(f"{Colors.YELLOW}Exiting conversation...{Colors.ENDC}")
                break
            
            elif user_input.lower() == 'help':
                print_help()
                continue
            
            elif user_input.lower() == 'state':
                print_state(agent)
                continue
            
            elif user_input.lower() == 'history':
                print_conversation_history(agent)
                continue
            
            elif user_input.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                print_header()
                continue
            
            # Process message through agent
            try:
                print(f"   {Colors.CYAN}[Processing...]{Colors.ENDC}")
                response = agent.chat(user_input)
                
                # Print response
                print(f"\n{Colors.GREEN}🤖 Agent: {Colors.ENDC}{response}\n")
                
                # Show state (compact)
                state = agent.get_state()
                print(f"{Colors.BOLD}[State]{Colors.ENDC} ", end="")
                print(f"Turn: {state['turn_count']} | ", end="")
                print(f"Intent: {Colors.YELLOW}{state['intent']}{Colors.ENDC} | ", end="")
                print(f"Lead: {Colors.GREEN if state['lead_captured'] else Colors.RED}{state['lead_captured']}{Colors.ENDC}\n")
                
            except Exception as e:
                print(f"{Colors.RED}❌ Error: {str(e)[:100]}{Colors.ENDC}")
                print(f"{Colors.YELLOW}(API rate limit or connection issue. Please try again later.){Colors.ENDC}\n")
                continue
    
    except Exception as e:
        print(f"{Colors.RED}❌ Unexpected error: {e}{Colors.ENDC}")
    
    finally:
        # Print summary and save
        print_summary(agent)
        
        # Ask to save conversation
        try:
            save_choice = input(f"{Colors.BOLD}Save conversation? (y/n): {Colors.ENDC}").strip().lower()
            if save_choice == 'y':
                save_conversation(agent)
        except (EOFError, KeyboardInterrupt):
            pass
        
        print(f"{Colors.GREEN}✅ Thanks for chatting with AutoStream!{Colors.ENDC}\n")


if __name__ == "__main__":
    chat_loop()