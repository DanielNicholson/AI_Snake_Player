import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    #plot last 25 scores
    plt.plot(scores[-100:], label='Score')
    #plot last 25 mean scores
    plt.plot(mean_scores[-100:], label='Mean Score')
    plt.title(label='Snake Game')
    plt.xlabel('number of games')
    plt.legend()
    plt.show()
    plt.pause(0.0001)

