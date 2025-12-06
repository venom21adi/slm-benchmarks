import ollama
import pandas as pd
import time
import datetime
import psutil
import os

# --- 1. Configuration ---
MODELS = [
    "gemma:2b",
    "mistral:7b-instruct", # Use the instruction variant for better QA
    "phi3:mini"
]
TEST_INPUT_TEXT = """
In the latest company report, we found that Jane Doe, the lead Software Engineer,
earns £85,000 annually. Her colleague, John Smith, a Senior Product Manager in Berlin,
has a yearly compensation of €95,000. Finally, Alex Chen, a Data Scientist based
in New York, is listed with a $130,000 salary.
"""

TEST_PROMPT = f"""
Analyze the following text and extract the names, their job title, and their reported salary.
Convert all salaries to USD. Assume the following conversion rates:
£1.00 = $1.25 USD
€1.00 = $1.08 USD

Output the result as a single JSON array of objects with the following keys: "name", "title", "salary_usd".

TEXT:
---
{TEST_INPUT_TEXT}
---
"""

NUM_PREDICT = 1000  # Target number of tokens to generate for throughput test
NUM_RUNS = 4        # Number of times to run each model/prompt combination

# --- 2. Setup and Client ---
client = ollama.Client(host='http://localhost:11434') # Ensure Ollama is running on this host

# Prepare output file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"ollama_benchmark_results_{timestamp}.csv"
RESULTS = []

print(f"Starting automated benchmark. Results will be saved to {LOG_FILE}\n")

# --- 3. Hardware Monitoring Function ---
def get_hardware_usage(process_name="ollama"):
    """
    Captures the current CPU and RAM usage for the Ollama process.
    """
    cpu_percent = 0
    mem_percent = 0
    
    # Iterate over all running processes
    for proc in psutil.process_iter(['name', 'cpu_percent', 'memory_info']):
        # Ollama often runs as a single process named 'ollama' or 'ollama.exe'
        if proc.info['name'].lower().startswith(process_name):
            try:
                # Use psutil to get CPU/Memory usage
                cpu_percent = proc.cpu_percent(interval=None) # Non-blocking
                mem_bytes = proc.memory_info().rss
                mem_percent = (mem_bytes / (1024**3)) # Convert to GB
                return cpu_percent, mem_percent
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    
    return 0, 0 # Return 0 if process not found

# --- 4. Main Testing Loop ---
for model in MODELS:
    print(f"--- Testing Model: {model} ({NUM_RUNS} runs) ---")
    
    for run_num in range(1, NUM_RUNS + 1):
        print(f"  > Run {run_num}/{NUM_RUNS}...")
        
        # 1. Warm-up (Important for Ollama's keep_alive)
        if run_num == 1:
             print("    Warming up...")
             client.generate(model=model, prompt="hi", stream=False, options={'num_predict': 1})
        
        # 2. Start Hardware Monitor (Initial reading)
        initial_cpu, initial_ram = get_hardware_usage()
        
        # 3. Time the API call (Cold-Start & Generation)
        start_time = time.time()
        
        try:
            # Use /api/generate without streaming to get final statistics in one JSON object
            response = client.generate(
                model=model,
                prompt=TEST_PROMPT,
                stream=False, # Wait for the full response to get final metrics
                options={
                    'num_predict': NUM_PREDICT,
                    'temperature': 0.1 # Keep output deterministic
                },
                # Unload model after test to simulate cold start for next run/model
                keep_alive='0s' 
            )
            
            end_time = time.time()
            
            # 4. Final Hardware Monitor (Peak/End reading - though real peak needs continuous monitoring)
            final_cpu, final_ram = get_hardware_usage()

            # 5. Extract Ollama Performance Metrics (returned in the response when stream=False)
            total_duration_ms = response.get('total_duration', 0) / 1e6 # ns to ms
            load_duration_ms = response.get('load_duration', 0) / 1e6 # ns to ms
            eval_duration_ms = response.get('eval_duration', 0) / 1e6 # ns to ms
            eval_count = response.get('eval_count', 0) # Output tokens generated
            
            # --- Calculated Metrics ---
            eval_rate = (eval_count / (eval_duration_ms / 1000)) if eval_duration_ms else 0 # tokens/s
            time_to_first_token_ms = load_duration_ms # Approximation using load time
            
            # 6. Log Results
            run_data = {
                'Timestamp': datetime.datetime.now().isoformat(),
                'Model': model,
                'Run_Number': run_num,
                'Prompt_Length_Tokens': response.get('prompt_eval_count', 0),
                'Output_Tokens_Count': eval_count,
                'Time_Total_s': round(end_time - start_time, 2),
                
                # Ollama Metrics
                'Ollama_Load_Duration_ms': round(load_duration_ms, 2),
                'Ollama_Eval_Duration_ms': round(eval_duration_ms, 2),
                'Ollama_Total_Duration_ms': round(total_duration_ms, 2),
                'Tokens_Per_Second': round(eval_rate, 2),
                
                # Hardware Metrics (PSUtil snapshot)
                'PSUtil_Initial_CPU_p': round(initial_cpu, 2),
                'PSUtil_Final_CPU_p': round(final_cpu, 2),
                'PSUtil_Initial_RAM_GB': round(initial_ram, 2),
                'PSUtil_Final_RAM_GB': round(final_ram, 2),

                # Output Snapshot
                'Response': response.get('response', '').replace('\n', ' '),
                
                
                # # Output Snapshot
                # 'Response_Start': response.get('response', '')[:100].replace('\n', ' ') + '...',
            }
            RESULTS.append(run_data)
            
            print(f"    - TTFT: {run_data['Ollama_Load_Duration_ms']:.2f} ms | Eval Rate: {run_data['Tokens_Per_Second']:.2f} tokens/s | RAM: {run_data['PSUtil_Final_RAM_GB']:.2f} GB")

        except Exception as e:
            print(f"    - ERROR: Failed to run {model} (Run {run_num}): {e}")
            error_data = {
                'Timestamp': datetime.datetime.now().isoformat(),
                'Model': model,
                'Run_Number': run_num,
                'Error': str(e)
            }
            RESULTS.append(error_data)

# --- 5. Save Results ---
df = pd.DataFrame(RESULTS)
df.to_csv(LOG_FILE, index=False)
print(f"\n✅ Benchmark complete. Results saved to: {os.path.abspath(LOG_FILE)}")