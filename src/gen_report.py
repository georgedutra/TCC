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

DIFFICULTY_LEVEL = 'easy'
MODELS = [
    "mistral:7b",
    "mistral:7b-text-q8_0",
    "mistral:7b-text-q4_0"
]

CHECKPOINT_INTERVAL = 10  # Save every 10 rows
CHECKPOINT_DIR = os.path.join(root, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
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

def process_model(model_name, df_questions, difficulty):
    """Process a single model with checkpoint support and error handling."""
    print(f"\n{'='*60}")
    print(f"Starting benchmark for {difficulty} difficulty with {model_name}")
    print(f"{'='*60}")
    
    # Try to load checkpoint
    df_checkpoint, metadata = load_checkpoint(model_name, difficulty)
    
    if df_checkpoint is not None:
        df_results = df_checkpoint
        start_index = metadata['last_processed_index'] + 1
    else:
        df_results = df_questions.copy()
        df_results[f'{model_name}_answer'] = ''
        df_results[f'{model_name}_time'] = 0.0
        start_index = 0
    
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
                    response = mdl.rag_respond(question, model_name, 1)
                    
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
    print(f"Starting benchmark suite for {DIFFICULTY_LEVEL} difficulty")
    print(f"Models to test: {', '.join(MODELS)}")
    
    # Load and filter dataset
    df = pd.read_csv(os.path.join(root, './data/evaluation_dataset.csv'))

    # Group by 'id' and filter by number of queries related
    if DIFFICULTY_LEVEL == 'easy':
        df_filtered = df[df['id'].apply(lambda x: str(x).split(',')).str.len() == 3]
    elif DIFFICULTY_LEVEL == 'medium':
        df_filtered = df[df['id'].apply(lambda x: str(x).split(',')).str.len() == 3]
    elif DIFFICULTY_LEVEL == 'hard':
        df_filtered = df[df['id'].apply(lambda x: str(x).split(',')).str.len() == 3]
    df_questions = df_filtered.head(100)
    
    print(f"Dataset size: {len(df_questions)} questions\n")
    # Process each model
    df_results = df_questions.copy()
    for model in MODELS:
        try:
            result = process_model(model, df_results, DIFFICULTY_LEVEL)
            if result is not None:
                df_results = result
        except KeyboardInterrupt:
            print("\n⚠ Benchmark interrupted by user. Progress saved.")
            break
        except Exception as e:
            print(f"\n✗ Skipping {model} due to error: {e}")
            continue
    
    i = 0
    # Save final combined results
    while os.path.exists(os.path.join(root, f'results_{DIFFICULTY_LEVEL}_{i}.csv')):
        i += 1

    output_path = os.path.join(root, f'results_{DIFFICULTY_LEVEL}_{i}.csv')
    df_results.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"✓ Final results saved to: {output_path}")
    print(f"{'='*60}")