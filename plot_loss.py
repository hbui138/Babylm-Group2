import os
import ast
import glob
import matplotlib.pyplot as plt

def generate_loss_charts(log_folder="logs", image_folder="images"):
    # Ensure the output directory exists
    os.makedirs(image_folder, exist_ok=True)
    
    # Define the search pattern for .out files
    search_pattern = os.path.join(log_folder, "*.out")
    
    # Iterate through all matching files
    for filepath in glob.glob(search_pattern):
        epochs = []
        losses = []
        
        # Variables to track phase resets and make epochs continuous
        cumulative_offset = 0.0
        last_epoch = 0.0
        
        # Read the file line by line
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                
                # Check if line contains dictionary format with loss data
                if line.startswith("{") and "'loss':" in line:
                    try:
                        # Safely evaluate the string to a dictionary
                        log_data = ast.literal_eval(line)
                        
                        # Extract epoch and loss values
                        if "epoch" in log_data and "loss" in log_data:
                            current_epoch = float(log_data["epoch"])
                            
                            # Detect if the epoch resets (e.g., drops from 10 to 0.015)
                            if current_epoch < last_epoch:
                                # Add the peak of the previous phase to the offset
                                cumulative_offset += last_epoch
                            
                            # Update last_epoch for the next iteration's comparison
                            last_epoch = current_epoch
                            
                            # Calculate the continuous epoch value
                            actual_epoch = current_epoch + cumulative_offset
                            
                            epochs.append(actual_epoch)
                            losses.append(float(log_data["loss"]))
                    except (ValueError, SyntaxError):
                        # Skip malformed lines
                        pass
        
        # If valid data was found, generate the plot
        if epochs and losses:
            filename = os.path.basename(filepath)
            base_name = os.path.splitext(filename)[0]
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, losses, color="#1f77b4", linewidth=2)
            
            # Formatting the chart
            plt.title(f"Training Loss over Time: {base_name}", fontsize=14, fontweight="bold")
            plt.xlabel("Continuous Epoch", fontsize=12) # Updated label
            plt.ylabel("Loss", fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.7)
            
            # Save the plot
            output_path = os.path.join(image_folder, f"{base_name}.png")
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close()
            
            print(f"Generated chart: {output_path}")

if __name__ == "__main__":
    generate_loss_charts()