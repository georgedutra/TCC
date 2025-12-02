# Adiciona o diretório raiz ao sys.path
import sys
import os
root = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.append(root)

import warnings
warnings.filterwarnings("ignore")

import tqdm
import pandas as pd
import json
import traceback
from datetime import datetime
from baseline_model import model as mdl
from utils.monitor import ResourceMonitor 

MODELS = [
    "mistral:7b",
    "mistral:7b-q8",
    "mistral:7b-fp16"
]

CHECKPOINT_INTERVAL = 10  # Save every 10 rows
CHECKPOINT_DIR = os.path.join(root, 'checkpoints')
REPORTS_DIR = os.path.join(root, 'reports')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(os.path.join(root, 'log'), exist_ok=True)


def get_checkpoint_path(model_name, difficulty):
    """Get checkpoint file path for a specific model and difficulty."""
    safe_model_name = model_name.replace(':', '_').replace('/', '_')
    return os.path.join(CHECKPOINT_DIR, f'checkpoint_{safe_model_name}_{difficulty}.csv')

def get_metadata_path(model_name, difficulty):
    """Get metadata file path for tracking progress."""
    safe_model_name = model_name.replace(':', '_').replace('/', '_')
    return os.path.join(CHECKPOINT_DIR, f'metadata_{safe_model_name}_{difficulty}.json')

def load_checkpoint(model_name, difficulty):
    """Load checkpoint if exists, return None otherwise."""
    checkpoint_path = get_checkpoint_path(model_name, difficulty)
    metadata_path = get_metadata_path(model_name, difficulty)
    
    if os.path.exists(checkpoint_path) and os.path.exists(metadata_path):
        try:
            df = pd.read_csv(checkpoint_path)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"✓ Resuming from checkpoint: row {metadata['last_processed_index'] + 1}")
            return df, metadata
        except Exception as e:
            print(f"⚠ Failed to load checkpoint: {e}")
            return None, None
    return None, None

def save_checkpoint(df, model_name, difficulty, last_index, total_rows):
    """Save checkpoint with metadata."""
    checkpoint_path = get_checkpoint_path(model_name, difficulty)
    metadata_path = get_metadata_path(model_name, difficulty)
    
    try:
        df.to_csv(checkpoint_path, index=False)
        metadata = {
            'last_processed_index': last_index,
            'total_rows': total_rows,
            'timestamp': datetime.now().isoformat(),
            'model': model_name,
            'difficulty': difficulty
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"⚠ Failed to save checkpoint: {e}")

def delete_checkpoints(model_name, difficulty):
    """Delete checkpoint files after successful completion."""
    checkpoint_path = get_checkpoint_path(model_name, difficulty)
    metadata_path = get_metadata_path(model_name, difficulty)
    
    try:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        print(f"✓ Checkpoints cleaned up for {model_name}")
    except Exception as e:
        print(f"⚠ Failed to delete checkpoints: {e}")

def get_results_path(difficulty, version=None):
    """Get the path for the results file.
    
    Args:
        difficulty: The difficulty level.
        version: Optional version number. If None, returns the base path without version.
    
    Returns:
        Path to the results file.
    """
    if version is None:
        return os.path.join(REPORTS_DIR, f'results_{difficulty}.csv')
    else:
        return os.path.join(REPORTS_DIR, f'results_{difficulty}_v{version}.csv')

def find_latest_results(difficulty):
    """Find the latest results file (highest version number).
    
    Returns:
        tuple: (dataframe, version) or (None, 0) if no results found.
    """
    # Check for unversioned file first
    base_path = get_results_path(difficulty)
    if os.path.exists(base_path):
        try:
            df = pd.read_csv(base_path)
            print(f"✓ Found existing results file: {base_path}")
            return df, 0
        except Exception as e:
            print(f"⚠ Failed to load base results file: {e}")
    
    # Look for versioned files
    import glob
    pattern = os.path.join(REPORTS_DIR, f'results_{difficulty}_v*.csv')
    versioned_files = glob.glob(pattern)
    
    if versioned_files:
        # Extract version numbers and find the highest
        versions = []
        for filepath in versioned_files:
            try:
                # Extract version number from filename
                basename = os.path.basename(filepath)
                version_str = basename.split('_v')[1].replace('.csv', '')
                versions.append((int(version_str), filepath))
            except:
                continue
        
        if versions:
            versions.sort(reverse=True)
            latest_version, latest_path = versions[0]
            try:
                df = pd.read_csv(latest_path)
                print(f"✓ Found existing results file: {latest_path} (version {latest_version})")
                return df, latest_version
            except Exception as e:
                print(f"⚠ Failed to load results file v{latest_version}: {e}")
    
    return None, 0

def load_existing_results(difficulty):
    """Load existing results file if it exists."""
    df, version = find_latest_results(difficulty)
    return df

def get_next_results_path(difficulty):
    """Get the path for saving new results (next version).
    
    Returns:
        Path for the new results file with incremented version.
    """
    _, current_version = find_latest_results(difficulty)
    next_version = current_version + 1
    return get_results_path(difficulty, next_version)

def get_completed_models(df, model_list):
    """Check which models have already been processed in the results dataframe."""
    completed = []
    pending = []
    
    for model in model_list:
        answer_col = f'{model}_answer'
        time_col = f'{model}_time'
        
        # Check if columns exist and have non-empty values
        if answer_col in df.columns and time_col in df.columns:
            # Check if at least 90% of rows are filled (to handle partial completions)
            filled_ratio = (df[answer_col].notna() & (df[answer_col] != '')).sum() / len(df)
            if filled_ratio >= 0.9:
                completed.append(model)
                print(f"  ✓ {model}: Already completed ({filled_ratio*100:.1f}% filled)")
            else:
                pending.append(model)
                print(f"  ⚠ {model}: Partially completed ({filled_ratio*100:.1f}% filled) - will reprocess")
        else:
            pending.append(model)
            print(f"  ○ {model}: Not started")
    
    return completed, pending

def process_model(model_name, df_results, difficulty):
    """Process a single model with checkpoint support and error handling."""
    print(f"\n{'='*60}")
    print(f"Starting benchmark for {difficulty} difficulty with {model_name}")
    print(f"{'='*60}")
    
    # Try to load checkpoint first
    df_checkpoint, metadata = load_checkpoint(model_name, difficulty)
    
    if df_checkpoint is not None:
        df_results = df_checkpoint
        start_index = metadata['last_processed_index'] + 1
    else:
        # Initialize columns if they don't exist
        if f'{model_name}_answer' not in df_results.columns:
            df_results[f'{model_name}_answer'] = ''
        if f'{model_name}_time' not in df_results.columns:
            df_results[f'{model_name}_time'] = 0.0
        
        # Check if there's existing data to resume from
        answer_col = f'{model_name}_answer'
        filled_mask = (df_results[answer_col].notna() & (df_results[answer_col] != ''))
        if filled_mask.any():
            # Find the last filled row
            last_filled = filled_mask[::-1].idxmax()
            start_index = df_results.index.get_loc(last_filled) + 1
            print(f"✓ Resuming from existing results: row {start_index}")
        else:
            start_index = 0
    
    if difficulty == 'easy': k = 1
    elif difficulty == 'medium': k = 2
    else: k = 3

    monitor = ResourceMonitor()
    monitor.start()
    
    try:
        # Lazy load the model
        mdl.get_llm(model_name=model_name, overwrite=True)
        mdl.respond("Hello", model_name=model_name)  # Warm-up call
        
        # Process rows
        indices = list(df_results.iterrows())
        with tqdm.tqdm(total=len(indices), initial=start_index, desc=f"{model_name}") as pbar:
            for idx, (index, row) in enumerate(indices):
                if idx < start_index:
                    continue
                
                try:
                    question = row['pergunta']
                    response = mdl.rag_respond(question, model_name, k)
                    
                    df_results.at[index, f'{model_name}_answer'] = response
                    df_results.at[index, f'{model_name}_time'] = mdl.rag_respond.last_execution_time
                    
                    monitor.sample()
                    pbar.update(1)
                    
                    # Save checkpoint every CHECKPOINT_INTERVAL rows
                    if (idx + 1) % CHECKPOINT_INTERVAL == 0:
                        save_checkpoint(df_results, model_name, difficulty, idx, len(indices))
                
                except Exception as e:
                    print(f"\n✗ Error processing row {idx}: {e}")
                    print(traceback.format_exc())
                    # Save checkpoint on error
                    save_checkpoint(df_results, model_name, difficulty, idx - 1, len(indices))
                    print(f"✓ Emergency checkpoint saved. You can resume from row {idx}")
                    raise  # Re-raise to stop processing this model
        
        # Save final results and monitoring data
        monitor.save_samples(os.path.join(root, f'log/{model_name}_{difficulty}_monitoring.csv'))
        monitor.plot(os.path.join(root, f'log/{model_name}_{difficulty}_monitoring.png'))
        
        # Delete checkpoints after successful completion
        # delete_checkpoints(model_name, difficulty)
        
        return df_results
        
    except KeyboardInterrupt:
        print(f"\n⚠ Interrupted by user. Checkpoint saved.")
        raise
    except Exception as e:
        print(f"\n✗ Fatal error processing {model_name}: {e}")
        print(traceback.format_exc())
        return None
    finally:
        # Always try to save monitoring data
        try:
            monitor.save_samples(os.path.join(root, f'log/{model_name}_{difficulty}_monitoring_partial.csv'))
        except:
            pass


# Main execution
if __name__ == "__main__":
    for difficulty in ['easy', 'medium', 'hard']:
        print(f"{'='*60}")
        print(f"BENCHMARK SUITE - {difficulty.upper()} DIFFICULTY")
        print(f"{'='*60}")
        
        # Load and filter dataset
        df = pd.read_csv(os.path.join(root, './data/evaluation_dataset.csv'))

        # Group by 'id' and filter by number of queries related
        if difficulty == 'easy':
            df_filtered = df[df['id'].apply(lambda x: str(x).split(',')).str.len() == 1] # Easy: single query
        elif difficulty == 'medium':
            df_filtered = df[df['id'].apply(lambda x: str(x).split(',')).str.len() == 2] # Medium: two related queries
        elif difficulty == 'hard':
            df_filtered = df[df['id'].apply(lambda x: str(x).split(',')).str.len() >= 3] # Hard: three or more related queries
    
        # Limit to first 100 questions for benchmarking
        df_questions = df_filtered.head(100)
        
        print(f"\nDataset size: {len(df_questions)} questions")
        print(f"Models to test: {', '.join(MODELS)}")
        print(f"Checkpoint interval: every {CHECKPOINT_INTERVAL} rows\n")
        
        # Check for existing results file
        df_existing = load_existing_results(difficulty)
        _, current_version = find_latest_results(difficulty)
        
        if df_existing is not None:
            print(f"\nChecking progress for each model:")
            completed_models, pending_models = get_completed_models(df_existing, MODELS)
            
            if len(completed_models) == len(MODELS):
                print(f"\n✓ All models already completed in current results file!")
                print(f"Starting a new run will create a new versioned results file.\n")
                response = input("Do you want to start a fresh run? (y/N): ").strip().lower()
                if response != 'y':
                    print("Exiting. No changes made.")
                    sys.exit(0)
                # Start fresh with new version
                print("Starting fresh run with new version...")
                models_to_process = MODELS
                df_results = df_questions.copy()
            elif completed_models:
                print(f"\n✓ Skipping {len(completed_models)} completed model(s): {', '.join(completed_models)}")
                models_to_process = pending_models
                df_results = df_existing
            else:
                models_to_process = MODELS
                df_results = df_existing
        else:
            print("No existing results found. Starting from scratch.\n")
            models_to_process = MODELS
            df_results = df_questions.copy()
        
        # Determine output path (will be versioned if this is a fresh run)
        if df_existing is None or (df_existing is not None and len(completed_models) == len(MODELS)):
            # New run or fresh run after completion - use next version
            output_path = get_next_results_path(difficulty)
            print(f"Results will be saved to: {output_path} (new version)")
        else:
            # Continuing existing run - keep same path
            if current_version == 0:
                output_path = get_results_path(difficulty)
            else:
                output_path = get_results_path(difficulty, current_version)
            print(f"Results will be updated in: {output_path}")
        
        # Process each pending model
        if models_to_process:
            print(f"\nProcessing {len(models_to_process)} model(s): {', '.join(models_to_process)}\n")
            
            for model in models_to_process:
                try:
                    result = process_model(model, df_results, difficulty)
                    if result is not None:
                        df_results = result
                        # Save intermediate results after each model
                        df_results.to_csv(output_path, index=False)
                        print(f"✓ Intermediate results saved to: {output_path}")
                except KeyboardInterrupt:
                    print("\n⚠ Benchmark interrupted by user. Progress saved.")
                    df_results.to_csv(output_path, index=False)
                    print(f"✓ Progress saved to: {output_path}")
                    break
                except Exception as e:
                    print(f"\n✗ Skipping {model} due to error: {e}")
                    # Save progress even on error
                    df_results.to_csv(output_path, index=False)
                    continue
        
        # Save final combined results
        df_results.to_csv(output_path, index=False)
        print(f"\n{'='*60}")
        print(f"✓ Final results saved to: {output_path}")
        print(f"{'='*60}")