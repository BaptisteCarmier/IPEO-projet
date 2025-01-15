
import pandas as pd
import matplotlib.pyplot as plt

def plot_final_graphs(csv_files, wd, lr):
    """
    Function to generate plots for each provided CSV file, with specified weight decay (wd) and learning rate (lr) parameters.

    Parameters:
        csv_files (list of str): List of file paths to CSV files.
        wd (float): Weight decay parameter to filter data.
        lr (float): Learning rate parameter to filter data.

    Returns:
        None
    """
    # Define the plot titles corresponding to each CSV file
    titles = ["12 Bands (Augmented)", "12 Bands (Not Augmented)", "3 Bands (Augmented)", "3 Bands (Not Augmented)"]

    if len(csv_files) != 4:
        raise ValueError("Expected exactly 4 CSV files.")

    # Create a single figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, csv_file in enumerate(csv_files):
        try:
            # Load the CSV file into a DataFrame
            df = pd.read_csv(csv_file)

            # Filter the DataFrame based on the wd and lr parameters
            filtered_df = df[(df['Weight Decay'] == wd) & (df['Learning Rate'] == lr)]

            if filtered_df.empty:
                print(f"No data found in {csv_file} for Weight Decay={wd} and Learning Rate={lr}")
                continue

            # Plot the data (assuming 'Epoch', 'Val RootLoss', and 'Train RootLoss' are columns in the data)
            ax = axes[i]
            ax.plot(filtered_df['Epoch'], filtered_df['Val Loss'], marker='o', label='Validation Loss')
            ax.plot(filtered_df['Epoch'], filtered_df['Train Loss'], marker='s', label='Training Loss')

            # Adding plot details
            ax.set_title(titles[i])
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Root Loss')
            ax.legend()
            ax.grid(True)

            # Add wd and lr text below each subplot
            ax.text(0.5, -0.14, f"Weight Decay: {wd}, Learning Rate: {lr}", transform=ax.transAxes, 
                    ha='center', fontsize=10, color='gray')

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

        

    # Adjust layout and show the figure
    plt.tight_layout()

     # Save the figure
    plot_filename = f"TrainGraphs"
    plt.savefig(plot_filename, dpi=300)

    plt.show()


# Example usage
if __name__ == "__main__":
    csv_files = [
        "1001_Run2_3bands_Aug.csv",
        "1001_Run2_3bands_NotAug.csv",
        "1001_Run2_12bands_Aug.csv",
        "1001_Run2_3bands_NotAug.csv"
    ]

    # Specify wd and lr parameters
    weight_decay = 1e-5
    learning_rate = 0.001

    # Generate plots
    plot_final_graphs(csv_files, weight_decay, learning_rate)