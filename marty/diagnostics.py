from typing import Iterable
from marty.actions import Action, Context
from torch.utils.tensorboard.writer import SummaryWriter
import matplotlib.pyplot as plt

writer = SummaryWriter()
_STEP = 0


def get_global_step():
    return _STEP


def increase_global_step():
    global _STEP
    _STEP += 1
    return _STEP


def display_action_scores(actions: Iterable[Action], scores, c: Context):
    fig = plt.figure(figsize=(8, 4), dpi=30)
    plt.bar(range(len(scores)), scores.detach().numpy())
    plt.xticks(range(len(scores)), [str(a)[7:-1] for a in actions], rotation=90)
    plt.title(str(c))
    plt.tight_layout()
    writer.add_figure("action", plt.gcf(), global_step=get_global_step())


def display_state(state_tens, c):
    fig = plt.figure(figsize=(8, 4), dpi=30)
    plt.bar(range(len(state_tens)), state_tens.detach().numpy())
    plt.title(str(c))
    plt.tight_layout()
    writer.add_figure("state", plt.gcf(), global_step=get_global_step())
