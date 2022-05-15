import numpy as np
import matplotlib.pyplot as plt

def main(r1, r2,title, vals, ax, ii):
    
    ax[ii].set_title(title)
    ax[ii].set_thetagrids(range(0, 360, int(360/len(methods))), (methods))
    ax[ii].yaxis.set_visible(False)

    ax2 = polar_twin(ax[ii])

    ax[ii].plot(theta, r2, color='blue', alpha=0.5, linewidth=2.5)
    ax2.plot(theta, r1, color='green', alpha=0.5, linewidth=2.5)

    ax2.set_ylim(vals[0], vals[1])
    ax[ii].set_ylim(vals[2], vals[3])
    
    return

def polar_twin(ax):
    ax2 = ax.figure.add_axes(ax.get_position(), projection='polar', 
                             label='twin', frameon=False,
                             theta_direction=ax.get_theta_direction(),
                             theta_offset=ax.get_theta_offset())
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(True)
    ax2.set_yticklabels([])

    # There should be a method for this, but there isn't... Pull request?
    # ax2._r_label_position._t = (22.5 + 180, 360)
    # ax2._r_label_position.invalidate()

    # Bit of a hack to ensure that the original axes tick labels are on top of
    # whatever is plotted in the twinned axes. Tick labels will be drawn twice.
    # for label in ax.get_yticklabels():
    #     ax.figure.texts.append(label)

    return ax2

if __name__ == '__main__':
    
    dice_acdc = [0.828,0.86,0.854,0.875,0.818,0.893,0.828]
    dice_mrbrain = [0.788,0.755,0.787,0.748,0.745,0.812,0.788]
    dice_brats = [0.775,0.75,0.747,0.765,0.781,0.769,0.775]
    dice_promise = [0.565,0.301,0.584,0.59,0.597,0.625,0.565]
    dice_refuge = [0.891,0.875,0.892,0.894,0.888,0.8868,0.891]
    dice_abdomen = [0.871, 0.876, 0.8807, 0.8885, 0.8786, 0.8855,0.871]
    
    rng_acdc = [0.65,1.0, 0.01, 0.28]
    rng_mrbrain = [0.60,0.9, 0.01, 0.20]
    rng_brats = [0.65,0.85, 0.01, 0.25]
    rng_promise = [0.25,0.75, 0.1, 0.7]
    rng_refuge = [0.8, 0.95, 0.01, 0.10]
    rng_abdomen = [0.8, 0.95, 0.01, 0.10]

    ece_acdc = [0.121,0.133,0.104,0.106,0.132,0.076,0.121]
    ece_mrbrain = [0.09, 0.105, 0.049, 0.037, 0.06, 0.051,0.09]
    ece_brats = [0.114,0.127,0.103,0.09,0.09,0.086,0.114]
    ece_promise = [0.294,0.469,0.288,0.211,0.195,0.241,0.294]
    ece_refuge = [0.032,0.04,0.037,0.06,0.029,0.033,0.032]
    ece_abdomen = [0.0492,0.0288,0.0452,0.0491,0.03289,0.0423,0.0492]
    
    methods = ['CE','FL','ECP','LS','SVLS','MbLS']
    
    theta = np.linspace(0, 2*np.pi, len(methods) + 1)
    params = dict(projection='polar', theta_direction=-1, theta_offset=np.pi/2)
    fig, ax = plt.subplots(1,6,subplot_kw=params, figsize=(25,25))
    
    main(dice_acdc, ece_acdc,'acdc',rng_acdc,ax, 0)
    main(dice_mrbrain, ece_mrbrain, 'mrbrain', rng_mrbrain,ax, 1)
    main(dice_brats, ece_brats, 'brats', rng_brats,ax, 2)
    main(dice_promise, ece_promise, 'promise', rng_promise,ax,3)
    main(dice_refuge, ece_refuge, 'refuge', rng_refuge,ax, 4)
    main(dice_abdomen, ece_abdomen, 'abdomen', rng_abdomen,ax, 5)
