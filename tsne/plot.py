from sklearn.manifold import TSNE
import os
import matplotlib.pyplot as plt

def tsne(penultimate, targets, config, perplexity):
    tsne = TSNE(perplexity=perplexity)
    tsne_results = tsne.fit_transform(penultimate)
    
    # Create the figure
    fig = plt.figure( figsize=(16,16) )
    ax = fig.add_subplot(1, 1, 1, title='t-SNE' )
    # Create the scatter
    scatter = ax.scatter(x=tsne_results[:,0], 
                    y=tsne_results[:,1], 
                    c=targets,
                    cmap=plt.cm.get_cmap('Paired'), 
                    alpha=0.4)
    legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower right", title="Classes")
    ax.add_artist(legend1)

    plt.tight_layout()
    plt.savefig(os.path.join(config.output_folder,
                '{0}+{1}_{2}epoch.png'.format(config.data, perplexity, config.best_epoch)), bbox_inches='tight')
    plt.show()