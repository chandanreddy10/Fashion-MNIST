import os
import matplotlib.pyplot as plt

def save_figure(fig, filename,fig_extension="png", resolution=300):
    """
    Save a Matplotlib figure to the 'figures' folder.

    Args:
        fig : The Matplotlib figure to save.
        filename: Name of the output file (e.g., 'plot.png').
    """
    # making sure output directory exists
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)

    # Save figure
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, format=fig_extension, dpi=resolution)
    print(f"Saved figure: {save_path}")
