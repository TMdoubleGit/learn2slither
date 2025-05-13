from matplotlib import pyplot as plt
from IPython import display
from src.constants import TRAINING_MODE
import os

plt.ion()

def plot(scores, mean_scores, save_path=None):
    # if not TRAINING_MODE:
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of games')
    plt.ylabel('Score')
    plt.plot(scores, label="Score")
    plt.plot(mean_scores, label="Mean Score", linestyle="dashed")
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.legend()
    # if not TRAINING_MODE:
    plt.pause(0.1)
    plt.show(block=False)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)