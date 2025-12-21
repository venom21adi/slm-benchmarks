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

# TEST_PROMPT = """
# Write a 500-word short story about a robot who discovers music. Focus on descriptive language and a clear narrative arc.
# """

# Summarization_text = """

# Wretched Little Victims of the Workhouses
# For George and Richard, the chocolate factory was to be much more than a
# commercial enterprise. As Quakers they shared a vision of social justice and
# reform: a new world in which the poor and needy would be lifted from the
# “ruin of deprivation.” For generations, Cadburys had been members of the
# Society of Friends or Quakers, a spiritual movement originally started by
# George Fox in the seventeenth century. In a curious irony, the very religion
# that inspired Quakers to act charitably towards the poor also produced a set
# of codes and practices that placed a few thousand close-knit families like the
# Cadburys in pole position to generate astounding material rewards at the
# start of the industrial age.
# Richard and George had been brought up on stories of George Fox, and
# many of the values, aspirations, and disciplines that shaped their lives
# stemmed from Fox's teachings. Born in 1624, the son of a weaver from
# Fenny Drayton, Leicestershire, Fox grew up with a passionate interest in
# religion at a time when the country had seen years of religious turmoil. Fox
# went “to many a priest looking for comfort, but found no comfort from
# them.” He was appalled at the inhumanity carried out in the name of
# religion: people imprisoned, hung, or even beheaded for their faith.
# Disregarding the danger following the outbreak of civil war in 1642, he left
# home the following year and set out on foot for London. At just nineteen
# years old, Fox embarked on a personal quest for greater understanding.
# During his years of travels, “when my hopes . . . in all men were gone,” he
# had an epiphany. The key to religion was not to be found in the sermons of
# preachers but in an individual's inner experience. Inspired, he began to speak
# out, urging people to listen to their own conscience. Because “God dwelleth
# in the hearts of obedient people,” he reasoned, it followed that an individual
# could find “the spirit of Christ within” to guide them, instead of taking
# orders from others. But his simple interpretation of Christianity put him in
# direct opposition to the authorities. If an individual was listening to the voice
# of God within himself, it followed that priests and religious authorities were
# a needless intermediary between man and God.
# Fox was perceived as dangerous and his preaching blasphemous to
# established churches. Even the like-minded Puritans objected. They too
# adhered to a rigorous moral code and high standards of self-discipline, and
# they disdained worldly pursuits. But Fox's emphasis on the direct
# relationship between a believer and God went far beyond what most Puritans
# deemed tolerable. In emphasizing the primary importance of an individual's
# experience, Fox appeared contemptuous of the authorities and mocked their
# petty regulations. For example, he would not swear on oath. If there was
# only one absolute truth, he reasoned, what was the point of a double
# standard, differentiating between “truth” and “truth on oath”?
# By 1649 Fox had crossed one magistrate too many. He was thrown into
# jail in Nottingham, “a pitiful stinking place, where the wind brought in all
# the stench of the house.” The following year he was jailed in Derby prison
# for blasphemy. A justice in Derby in 1650 is believed to be the first to use
# the term “Quaker” to mock George Fox and his followers. He scoffed at the
# idea expressed in their meetings in which they were “silent before God” until
# moved to speak, “trembling at the word of the God.” Despite its origins as a
# term of abuse, the name Quaker soon became widespread.
# Fox was imprisoned sixty times, but the Quaker movement continued to
# gain momentum. It is estimated that during the reign of Charles II, 198
# Quakers were transported overseas as slaves, 338 died from injuries received
# defending their faith, and 13,562 were imprisoned. Among them were
# Richard and George's forebears on their father's side, including Richard
# Tapper Cadbury, a “woolcomber” who was held in Southgate prison in
# Exeter in 1683 and again in 1684.
# By the end of Fox's life in 1691, there were 100,000 Quakers, and the
# movement had spread to America, parts of Europe, and even the West Indies.
# Fox established a series of meetings for Friends to discuss issues and
# formalize business: regional Monthly Meeting, county Quarterly Meeting,
# and a national Yearly Meeting. Key decisions at these meetings were written
# down and became known as the Advices. By 1738 these writings had been
# collated by clerks, transcribed in elegant longhand, and bound in a green
# manuscript, Christian and Brotherly Advices, which was made available to
# Friends Meetings across the country. It set out codes of personal conduct for
# Friends, under such headings as “Love,” “Covetousness,” and “Discipline.”
# A section on “Plainness,” for example, encouraged Quakers to cultivate
# “plainness of speech, behaviour and apparel.” A Friend's clothing should be
# dark and unadorned; even collars were removed from jackets as they were
# deemed too decorative.
# """

# TEST_PROMPT = f"""
# Write a 100-word summary for the text: {Summarization_text}

# """

# TEST_INPUT_TEXT = """
# In the latest company report, we found that Jane Doe, the lead Software Engineer,
# earns £85,000 annually. Her colleague, John Smith, a Senior Product Manager in Berlin,
# has a yearly compensation of €95,000. Finally, Alex Chen, a Data Scientist based
# in New York, is listed with a $130,000 salary.
# """

# TEST_PROMPT = f"""
# Analyze the following text and extract the names, their job title, and their reported salary.
# Convert all salaries to USD. Assume the following conversion rates:
# £1.00 = $1.25 USD
# €1.00 = $1.08 USD

# Output the result as a single JSON array of objects with the following keys: "name", "title", "salary_usd".

# TEXT:
# ---
# {TEST_INPUT_TEXT}
# ---
# """

TEST_PROJECT_DATA = """
Project: "Next-Gen E-Commerce Platform Launch"
Total Budget: $250,000 USD
Timeline Constraint: Must be completed in 10 weeks or less.

Available Tasks and Resources:
1. Database Schema Design (Priority: High) - Est. Cost: $40,000 - Est. Time: 2 weeks
2. API & Backend Development (Priority: High) - Est. Cost: $110,000 - Est. Time: 4 weeks
3. Mobile App Integration (Priority: Low) - Est. Cost: $95,000 - Est. Time: 3 weeks
4. Core Web UI/UX Design (Priority: Medium) - Est. Cost: $70,000 - Est. Time: 3 weeks
5. Cloud Infrastructure Setup (Priority: High) - Est. Cost: $30,000 - Est. Time: 1 week
6. Automated Testing Suite (Priority: Medium) - Est. Cost: $55,000 - Est. Time: 2 weeks
7. Security Audit & Penetration Testing (Priority: High) - Est. Cost: $60,000 - Est. Time: 2 weeks

Rules:
1. Priority 'High' tasks MUST be included.
2. The total budget MUST NOT exceed $250,000.
3. The total estimated time MUST NOT exceed 10 weeks.
4. If the total cost or time exceeds the limits, you must cut 'Low' and then 'Medium' priority tasks to meet the constraints.
5. If two tasks have the same priority and must be cut, cut the task with the highest cost first.
"""

TEST_PROMPT = f"""
You are an AI Project Manager. Your task is to analyze the following project data and constraints, and then create a final, executable project plan.

Perform the following steps:
1. **Analyze Constraints:** Determine which tasks are mandatory (High priority) and calculate their cumulative cost and time.
2. **Prioritization:** Determine which optional tasks (Medium/Low) must be kept or cut to fit the budget ($250,000) and time (10 weeks) constraints, strictly following the cutting rules.
3. **Generate Final Plan:** Create the definitive list of tasks to be included in the project.

Output your response in a single JSON object with the following three keys:
1. `initial_analysis`: A string detailing the mandatory tasks, their total cost, and total time, and the remaining budget/time.
2. `cut_tasks`: An array of strings listing ONLY the names of the tasks that were cut and the reason why (budget or time).
3. `final_plan`: An array of objects, one for each included task, with keys: "id" (1-7), "task", "cost", and "time". The tasks must be ordered by descending Priority (High, Medium, Low).

PROJECT DATA:
---
{TEST_PROJECT_DATA}
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