import matplotlib.pyplot as plt
import os

def visualize_loss_acc(train_losses, train_acc, val_losses, val_acc, title):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(title)  # Set overall
    ax[0, 0].plot(train_losses)
    ax[0, 0].set_title('Train Loss')
    ax[0, 1].plot(train_acc)
    ax[0, 1].set_title('Train Accuracy')
    ax[1, 0].plot(val_losses, 'orange')
    ax[1, 0].set_title('Validation Loss')
    ax[1, 1].plot(val_acc, 'orange')
    ax[1, 1].set_title('Validation Accuracy')
    plt.show()
    # Save the figure as a PNG file
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_file_directory, 'result')
    fig.savefig(os.path.join(path, title + '.png'))