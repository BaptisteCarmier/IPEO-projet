import matplotlib.pyplot as plt
import pandas as pd


# Set the CSV file name
csv_file = "1001_Run2_3bands_Aug"  # Change this to the name of your CSV file

def plot_results(csv_file):
    # Define column names (manually specify columns)
    column_names = [
        "Learning Rate", "Weight Decay", "Epoch", "Train Loss", "Train Loss",
        "Val Loss", "Val Loss"
    ]

    # Load the raw data from the CSV file into a Pandas DataFrame
    try:
        data = pd.read_csv(f"{csv_file}.csv")
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        return

    # Check if the required columns exist
    required_columns = ["Learning Rate", "Weight Decay", "Epoch", "Train Loss", "Train Loss", "Val Loss", "Val Loss", "Time Epoch"]
    for column in required_columns:
        if column not in data.columns:
            print(f"Error: Missing required column '{column}' in the CSV file.")
            return

    # Group data by Learning Rate
    grouped_by_lr = data.groupby("Learning Rate")
    
    # Iterate over each Learning Rate group
    for lr, group in grouped_by_lr:
        # Create a figure with 4 subplots (2 rows, 2 columns)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

        # Group further by Weight Decay within this Learning Rate
        grouped_by_wd = group.groupby("Weight Decay")
        for i, (wd, subgroup) in enumerate(grouped_by_wd):
            ax = axes[i]  # Select the subplot for this Weight Decay

            # Plot Train Loss and Val Loss for this Weight Decay
            ax.plot(subgroup["Epoch"], subgroup["Train Loss"], label="Train Loss", marker="o", color="blue")
            ax.plot(subgroup["Epoch"], subgroup["Val Loss"], label="Val Loss", marker="o", color="red")

            # Find the best validation Loss and corresponding epoch
            best_val_Loss = subgroup["Val Loss"].min()
            best_epoch = subgroup[subgroup["Val Loss"] == best_val_Loss]["Epoch"].iloc[0]

            # Find the last 20 epochs in the subgroup
            last_20_epochs = subgroup[subgroup["Epoch"] >= (subgroup["Epoch"].max() - 19)]

            # Calculate the average and variance of the Val Loss for the last 20 epochs
            avg_val_Loss = last_20_epochs["Val Loss"].mean()
            std_val_Loss = last_20_epochs["Val Loss"].std()

            # Annotate the best validation Loss, epoch, average, and variance
            ax.text(
                0.5, -0.35,  # Adjust the position as needed
                f"Best Val Loss = {best_val_Loss:.4f} at Epoch = {best_epoch}\n"
                f"Last 20 Epochs Avg Val Loss = {avg_val_Loss:.4f}\n"
                f"Last 20 Epochs Std Val Loss = {std_val_Loss:.4f}",
                fontsize=10,
                ha='center',
                transform=ax.transAxes
            )

            # Set subplot title and labels
            ax.set_title(f"Weight Decay = {wd}", fontsize=14)
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Loss", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True)



        # Add a global title below the file name
        fig.suptitle(f"{csv_file} \n Training and Validation Loss (Learning Rate = {lr})", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the global title

        # Save the figure
        plot_filename = f"{csv_file}_curve_lr{lr}".replace('.', '_')
        plt.savefig(plot_filename, dpi=300)

        # Show the plot
        plt.show()    

# Plot results
plot_results(csv_file)