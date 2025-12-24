import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import hydra
from pathlib import Path
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)
logger = logging.getLogger("DataViz")

from src.utils import setup_logging

def load_class_names(json_path: Path) -> dict:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {int(k): v for k, v in data.items()}
    except Exception as e:
        logger.warning(f"Could not load class names from {json_path}. Using IDs instead. Error: {e}")
        return {}
    
def load_client_data(cfg: DictConfig, project_root: Path) -> pd.DataFrame:
    """
    Scans the processed data directory for files matching 'client_*_train.pt'.
    """
    client_counts = {}

    try:
        data_dir = project_root / cfg.preprocess.output_dir
    except Exception:
        # Fallback if config structure is different
        data_dir = project_root / "data" / "processed"

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return pd.DataFrame()

    logger.info(f"Scanning for client data in: {data_dir}")

    client_files = sorted(list(data_dir.glob("client_*_train.pt")))

    if not client_files:
        logger.warning("No client training files found (pattern: client_*_train.pt).")
        return pd.DataFrame()

    for file_path in client_files:
        try:
            # Extract Client ID from filename (e.g., "client_10_train.pt" -> "10")
            parts = file_path.stem.split('_') # ['client', '10', 'train']
            if len(parts) >= 2 and parts[1].isdigit():
                client_id = parts[1]
            else:
                continue # Skip files that don't match the expected format

            data = torch.load(file_path, weights_only=True)
            labels = data['labels'].numpy()
            
            # Calculate frequency of each class label
            unique, counts = np.unique(labels, return_counts=True)
            client_counts[f"Client {client_id}"] = dict(zip(unique, counts))
            
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")

    logger.info(f"Successfully loaded data for {len(client_counts)} clients.")

    df = pd.DataFrame(client_counts).fillna(0).astype(int).T
    
    # Sort by Client ID (numeric) so Client 2 comes before Client 10
    try:
        df = df.sort_index(key=lambda x: x.map(lambda k: int(k.split()[-1])))
    except:
        pass 

    return df

def plot_stacked_bar(df: pd.DataFrame, class_map: dict, save_dir: Path):
    """
    Generates a Stacked Bar Chart to visualize class imbalance (Non-IID).
    """
    if class_map:
        df = df.rename(columns=class_map)
    
    # Sort for cleaner visualization
    df = df.sort_index(axis=1).sort_index(axis=0)
    
    # --- THE FIX: Normalize data to percentage (0 to 1) ---
    # We divide each row by the total samples in that row (client)
    df_percent = df.div(df.sum(axis=1), axis=0) * 100
    
    num_classes = len(df.columns)
    colors = sns.color_palette("husl", num_classes)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot the PERCENTAGE dataframe, not the raw counts
    df_percent.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=colors,
        width=0.95, 
        edgecolor='white',
        linewidth=0.5
    )

    ax.set_title('Dirichlet Data Partition (Class Distribution %)', fontweight='bold', fontsize=15)
    ax.set_ylabel('Percentage of Local Data (%)') # Changed label
    ax.set_xlabel('Client ID')
    
    # Move legend outside
    ax.legend(title='Classes', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    
    # Optional: Fix y-axis to strictly 0-100
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    
    output_path = save_dir / "dirichlet_distribution_percentage.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Normalized Plot saved: {output_path}")


def plot_heatmap(df: pd.DataFrame, class_map: dict, save_dir: Path):
    """
    Generates a Heatmap to inspect specific sample counts.
    Useful for debugging empty classes or extreme imbalances.
    """
    if class_map:
        df = df.rename(columns=class_map)
        
    df = df.sort_index(axis=1).sort_index(axis=0)

    plt.figure(figsize=(14, 8))
    
    sns.heatmap(
        df,
        annot=True,     # Write the actual number in the cell
        fmt="d",        # Format as integer (no decimals)
        cmap="YlGnBu",  # Color map: Yellow -> Green -> Blue
        linewidths=.5,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Client Class Distribution Heatmap', fontweight='bold', fontsize=16)
    plt.ylabel('')
    plt.xlabel('Class Labels')
    
    plt.tight_layout()
    
    output_path = save_dir / "dirichlet_distribution_heatmap.png"
    plt.savefig(output_path, dpi=300)
    logger.info(f"Heatmap saved: {output_path}")


@hydra.main(config_path="conf", config_name="config_fl", version_base=None)
def main(cfg: DictConfig):
    project_root = Path(get_original_cwd())

    log_file = project_root / "logs" / "visualize.log"
    log_level = str(cfg.get("log_level", "INFO")).upper()
    setup_logging(log_file_path=str(log_file), log_level=log_level)
    
    logger.info("---  Starting Visualization Pipeline ---")

    figs_dir = project_root / cfg.paths.figures_dir
    figs_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Visualization output directory: {figs_dir}")

    class_names_path = project_root / cfg.paths.class_names
    class_map = load_class_names(class_names_path)
    
    df_counts = load_client_data(cfg, project_root)
    
    if df_counts.empty:
        logger.warning("No client data was loaded. Please check 'paths' in config_fl.yaml.")
        return

    plot_stacked_bar(df_counts, class_map, figs_dir)
    plot_heatmap(df_counts, class_map, figs_dir)
    
    logger.info("---  Visualization Pipeline FINISHED ---")



if __name__ == "__main__":
    main()