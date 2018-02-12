    """
    f, axes = plt.subplots(1, 4, figsize=(8,1.5), dpi=400, sharey=True)
    colors = ["#98c37e", "#5788e7", "#D66667", "#B7B7B7"]
    for i in range(4):
        for col in ethoYdfs[i].columns:
            axes[i].plot(np.cumsum(ethoYdfs[i].loc[:,col]), color='#8c8c8c', alpha=0.2, lw=1)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
        axes[i].plot(np.mean(np.array(ethoYdfs[i])), color=colors[i], alpha=0.5, lw=1)
        if i>0:
            axes[i].spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(profile.out(), 'plots', 'cumsum_etho.png'), dpi=600)
    """
