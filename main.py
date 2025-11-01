import pickle
import os
from dotenv import load_dotenv
from gridworld import GridWorld
from q_learning import QLearningAgent
from agent_tools import AgentTools
from llm_agent import LLMAgent

# Load environment variables from .env file
load_dotenv()

def load_grid_config(filepath='models/grid_config.pkl'):
    """Load saved grid configuration"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Grid config not found at {filepath}. Run train_rl.py first!")
    
    with open(filepath, 'rb') as f:
        grid_data = pickle.load(f)
    
    return grid_data

def create_env_from_config(grid_data):
    """Create GridWorld environment from saved configuration"""
    env = GridWorld(size=grid_data['size'], num_obstacles=0)
    env.grid = grid_data['grid']
    env.start = grid_data['start']
    env.target = grid_data['target']
    env.current_pos = list(env.start)
    return env

def main():
    print("=" * 60)
    print("GRIDWORLD LLM AGENT - MAIN")
    print("=" * 60)
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("\n❌ Error: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Load grid configuration
    print("\n📁 Loading grid configuration...")
    try:
        grid_data = load_grid_config('models/grid_config.pkl')
        env = create_env_from_config(grid_data)
        print("✅ Grid loaded successfully")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("Please run 'python train_rl.py' first to train the model and save the grid.")
        return
    
    # Load Q-learning agent
    print("\n🤖 Loading Q-learning model...")
    try:
        q_agent = QLearningAgent()
        q_agent.load('models/q_table.pkl')
        print("✅ Q-learning model loaded successfully")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("Please run 'python train_rl.py' first to train the model.")
        return
    
    # Create agent tools
    print("\n🔧 Initializing agent tools...")
    tools = AgentTools(env, q_agent)
    print("✅ Tools initialized")
    print(f"   Available tools: {len(tools.get_available_tools())}")
    
    # Create LLM agent
    print("\n🧠 Initializing LLM agent (GPT-4o)...")
    llm_agent = LLMAgent(tools, env)
    print("✅ LLM agent ready")
    
    # Run the agent
    print("\n" + "=" * 60)
    print("STARTING GAME")
    print("=" * 60)
    
    result = llm_agent.run()
    
    # Summary
    print("\n" + "=" * 60)
    print("GAME SUMMARY")
    print("=" * 60)
    print(f"Success: {'✅ Yes' if result['success'] else '❌ No'}")
    print(f"Final Score: {result['score']}")
    print(f"Total Moves: {result['moves']}")
    print(f"LLM Iterations: {result['iterations']}")
    print("=" * 60)

if __name__ == "__main__":
    main()