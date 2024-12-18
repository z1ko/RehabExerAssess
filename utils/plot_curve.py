import os
import matplotlib.pyplot as plt


def plot_train_curve(movement,losses_epoch_train, losses_epoch_val, fig_dir):
    plt.figure(figsize=[15, 12], dpi=200)

    plt.subplot(1,2,1)
    plt.title(f'Loss {movement}')
    plt.plot(losses_epoch_train, 'b', label='Train loss')
    plt.plot(losses_epoch_val,'g', label='Val loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.title(f'Loss {movement} detail')
    plt.plot(losses_epoch_train, 'b', label='Train loss')
    plt.plot(losses_epoch_val,'g', label='Val loss')
    plt.ylim([0,20])
    plt.legend()

    plt.savefig(os.path.join(fig_dir, 'train_curve.jpg'))

    plt.clf()
    plt.close('all')
