import os
from openai import OpenAI

class LLMAgent:
    """LLM-powered agent that uses tools to navigate the gridworld"""
    
    def __init__(self, tools, env, api_key=None):
        """
        Initialize LLM agent
        
        Args:
            tools: AgentTools instance
            env: GridWorld environment
            api_key: OpenAI API key (if None, reads from environment)
        """
        self.tools = tools
        self.env = env
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.history = []
        self.max_iterations = 100
    
    def get_system_prompt(self):
        """Create system prompt for the LLM"""
        return """You are an intelligent agent navigating a gridworld to reach a target.

GRID LAYOUT:
- '.' = empty cell (penalty: -1)
- 'X' = obstacle cell (penalty: -5)
- 'A' = your current position
- 'T' = target (reward: +10)
- Grid coordinates: (row, column), starting from (0, 0) at top-left

YOUR GOAL:
Reach the target 'T' with the HIGHEST possible score. Avoid obstacles when possible!

AVAILABLE TOOLS:
- move_up/down/left/right: Execute a move
- best_move: Execute the best possible move
- random_move: Execute a random move
- rl_move: Use trained Q-learning model
- bfs_move: Use BFS shortest path algorithm
- look_ahead_all: See ALL 4 directions (ONLY ONCE)
- look_ahead_random: See ONE random direction
- get_current_state: Get current position and score

IMPORTANT LIMITS:
- look_ahead_all: Can use ONLY ONCE in the entire game!
- look_ahead_random: Can use ONLY ONCE per iteration!
- Use your one look_ahead_all wisely early in the game
- Use look_ahead_random when you need more info

STRATEGY TIPS:
- Avoid obstacles (X) - they give -5 penalty!
- BFS finds shortest path but doesn't avoid obstacles
- RL model was trained to maximize score
- Balance speed vs score

INSTRUCTIONS:
1. Analyze the current situation
2. Choose ONE tool to use
3. Explain your reasoning briefly
4. Format: [Your reasoning] TOOL: tool_name

Be strategic and maximize your score!"""
    
    def create_user_prompt(self, is_first=False):
        """Create user prompt with current game state"""
        state = self.tools.get_current_state()
        grid_visual = self.env.get_grid_string()
        
        if is_first:
            prompt = f"""GAME START!

Current Grid:
{grid_visual}

Current Position: {state['position']}
Target Position: {state['target']}
Current Score: {state['score']}
Moves Made: {state['moves']}

Which tool do you want to use? Explain your reasoning and then state your choice clearly.
Format: [Your reasoning] TOOL: tool_name"""
        else:
            prompt = f"""Current Grid:
{grid_visual}

Current Position: {state['position']}
Target Position: {state['target']}
Current Score: {state['score']}
Moves Made: {state['moves']}

Which tool do you want to use next?
Format: [Your reasoning] TOOL: tool_name"""
        
        return prompt
    
    def extract_tool_choice(self, response):
        """Extract tool name from LLM response"""
        # Look for "TOOL: tool_name" pattern
        if "TOOL:" in response:
            tool_part = response.split("TOOL:")[-1].strip()
            tool_name = tool_part.split()[0].strip()
            return tool_name
        
        # Fallback: check if any tool name is mentioned
        available_tools = [
            'move_up', 'move_down', 'move_left', 'move_right',
            'best_move', 'random_move', 'rl_move', 'bfs_move'
        ]
        
        response_lower = response.lower()
        for tool in available_tools:
            if tool in response_lower:
                return tool
        
        return None
    
    def execute_tool(self, tool_name):
        """Execute the chosen tool"""
        tool_methods = {
            'move_up': self.tools.move_up,
            'move_down': self.tools.move_down,
            'move_left': self.tools.move_left,
            'move_right': self.tools.move_right,
            'best_move': self.tools.best_move,
            'random_move': self.tools.random_move,
            'rl_move': self.tools.rl_move,
            'bfs_move': self.tools.bfs_move,
            'look_ahead_all': self.tools.look_ahead_all,
            'look_ahead_random': self.tools.look_ahead_random,
            'get_current_state': self.tools.get_current_state
        }
        
        if tool_name not in tool_methods:
            return {'error': f'Unknown tool: {tool_name}'}
        
        return tool_methods[tool_name]()
    
    def run(self):
        """Run the agent loop"""
        print("=" * 60)
        print("LLM AGENT PLAYING GRIDWORLD")
        print("=" * 60)
        
        # Initial state
        self.env.reset()
        print("\nInitial Grid:")
        self.env.render()
        
        # Prepare conversation
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": self.create_user_prompt(is_first=True)}
        ]
        
        iteration = 0
        done = False
        
        while iteration < self.max_iterations and not done:
            iteration += 1
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration}")
            print("=" * 60)
            
            # Reset per-iteration limits
            self.tools.reset_iteration_limits()
            
            # Get LLM response
            print("\nLLM is thinking...")
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            llm_response = response.choices[0].message.content
            print(f"\nLLM Response:\n{llm_response}")
            
            # Extract tool choice
            tool_name = self.extract_tool_choice(llm_response)
            
            if tool_name is None:
                print("\n❌ Could not extract tool choice from LLM response!")
                break
            
            print(f"\n🔧 Executing tool: {tool_name}")
            
            # Execute tool
            result = self.execute_tool(tool_name)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                break
            
            # Display result
            print(f"\n📊 Result:")
            for key, value in result.items():
                print(f"  {key}: {value}")
            
            # Show updated grid after move (if it was an actual move, not just look_ahead)
            move_tools = ['move_up', 'move_down', 'move_left', 'move_right', 
                         'best_move', 'random_move', 'rl_move', 'bfs_move']
            if tool_name in move_tools and 'error' not in result:
                print("\n🗺️  Updated Grid:")
                self.env.render()
            
            # Update conversation
            messages.append({"role": "assistant", "content": llm_response})
            
            # Check if done
            if result.get('done', False):
                done = True
                print(f"\n{'='*60}")
                print("🎉 TARGET REACHED!")
                print("=" * 60)
                print(f"\nFinal Score: {self.env.score}")
                print(f"Total Moves: {self.env.moves}")
                print("\nFinal Grid:")
                self.env.render()
                break
            
            # Continue conversation
            feedback = f"Tool executed: {tool_name}\nResult: {result}\n\nContinue to the next move."
            messages.append({"role": "user", "content": feedback})
        
        if not done:
            print(f"\n{'='*60}")
            print(f"❌ Did not reach target in {self.max_iterations} iterations")
            print("=" * 60)
            print(f"Final Score: {self.env.score}")
            print(f"Total Moves: {self.env.moves}")
        
        return {
            'success': done,
            'score': self.env.score,
            'moves': self.env.moves,
            'iterations': iteration
        }