"""
Plot figs along program flow (VISUAL)
"""
def plot_along(f, ax):
    warnings.filterwarnings("ignore")
    mng = plt.get_current_fig_manager()
    ### works on Ubuntu??? >> did NOT working on windows
# mng.resize(*mng.window.maxsize())
    mng.window.state('zoomed') #works fine on Windows!
    f.show()
    try:
        f.canvas.start_event_loop(0)
    except tk.TclError:
        pass
    warnings.filterwarnings("default")


"""
Plotting trajectory in arenas
"""
def plot_fly(data, x=None, y=None, hx=None, hy=None, etho=None, arena=None, spots=None, title=None, ax=None):
    spot_colors = {'yeast': '#ffc04c', 'sucrose': '#4c8bff'}
    ethocolor = {-1:'#7c7c7c', 0:'#404040', 1:'#7f00ac',2:'#ff6f6f', 3:'#4fff76', 4: '#ffc04c', 5: '#4c8bff', 6: '#ff0000'}
    inner_radius = arena['radius']/arena['scale']
    outer_radius = arena['outer']/arena['scale']
    if arena is not None:
        arena_border = plt.Circle((0, 0), inner_radius, color='k', fill=False)
        ax.add_artist(arena_border)
        outer_arena_border = plt.Circle((0, 0), outer_radius, color='#aaaaaa', fill=False)
        ax.add_artist(outer_arena_border)
        ax.plot(0, 0, 'o', color='black', markersize=2)
    if spots is not None:
        selected = [1, 8]
        for i, each_spot in enumerate(spots):
            substr = each_spot['substr']
            spot = plt.Circle((each_spot['x'], each_spot['y']), each_spot['r'], color=spot_colors[substr], alpha=0.5)
            if i in selected:
                ax.add_artist(plt.Circle((each_spot['x'], each_spot['y']), 2.5, facecolor='none', edgecolor='k', ls='--',lw=.75, alpha=0.5))
                ax.add_artist(plt.Circle((each_spot['x'], each_spot['y']), 5, facecolor='none', edgecolor='k', ls='--',lw=0.75, alpha=0.5))
            ax.add_artist(spot)
    ax.plot(data[x], data[y], 'k-', lw=0.5)
    #ax.plot(data[hx], data[hy], 'r-', lw=0.5)
    if etho is not None:
        for each_val in data[etho].unique():
            if each_val in [-1, 0, 1, 2, 3, 4, 5, 6]:
                this_x, this_y = data.query(etho+' == '+str(each_val))[hx],  data.query(etho+' == '+str(each_val))[hy]
                ax.scatter(this_x, this_y, c=ethocolor[each_val], s=10, alpha=1, marker='.')
                if each_val in [3]:
                    this_x, this_y = data.query(etho+' == '+str(each_val))[x],  data.query(etho+' == '+str(each_val))[y]
                    ax.scatter(this_x, this_y, c=ethocolor[each_val], s=20, alpha=1, marker='.')
    ax.scatter(np.array(data[x])[0], np.array(data[y])[0], c='#fffd11', s=50, alpha=1, marker='*')
    if arena is not None:
        ax.set_xlim([-1.1*outer_radius, 1.1*outer_radius])
        ax.set_ylim([-1.1*outer_radius, 1.1*outer_radius])
    ax.set_aspect("equal")
    return ax

        """
        totals['session'].append(each.name)
        totals['totals_Y'].append(ytotal)
        totals['totals_S'].append(stotal)
        totals['condition'].append(meta['fly']['metabolic'])
        max_frame = 108000
        ysignal = np.array(df['etho'])[:max_frame]
        ysignal[ysignal != 4] = 0
        ysignal[ysignal == 4] = 1
        ssignal = np.array(df['etho'])[:max_frame]
        ssignal[ssignal != 4] = 0
        ssignal[ssignal == 4] = 1

        if meta['fly']['metabolic'] == 'SAA':
            ethoY[0][each.name] = ysignal
            ethoS[0][each.name] = ssignal
        if meta['fly']['metabolic'] == 'AA':
            ethoY[1][each.name] = ysignal
            ethoS[1][each.name] = ssignal
        if meta['fly']['metabolic'] == 'S':
            ethoY[2][each.name] = ysignal
            ethoS[2][each.name] = ssignal
        if meta['fly']['metabolic'] == 'O':
            ethoY[3][each.name] = ysignal
            ethoS[3][each.name] = ssignal
    else:
        print('Excluded', each.name, meta['flags']['mistracked_frames'])

totals = pd.DataFrame(totals)
ethoYdfs = [pd.DataFrame(each) for each in ethoY]
ethoSdfs = [pd.DataFrame(each) for each in ethoS]
        """


## PLOTTING
"""
pals = {    'condition': {"SAA": "#98c37e", "AA": "#5788e7", "S":"#D66667", "O": "#B7B7B7"},
            'daytime': {8: '#445cff', 11: '#ffe11c', 14: '#ff9203', 17: '#992c03'},
            'day': None,
            'position': {"topleft": '#5F3C2B', "topright": '#AA6B46', "bottomleft": '#D3884F', "bottomright": '#DEAE95'}
        }
xlabels = {'condition': '',
           'daytime': 'Daytime',
           'day': 'Day',
           'position': 'Arena position'}
ylabels = {'totals_Y': 'Total yeast micromovements [min]',
           'totals_S': 'Total sucrose micromovements [min]',
           'abs_turn_rate': 'Mean absolute turning rate [º/s]',
           'distance': 'Distance travelled [mm]',
           'dcenter': 'Mean distance to center [mm]'}
cat = 'condition'
my_pal = pals[cat]
_order = None
_morder = None
if cat == 'condition':
    _order = ['SAA', 'AA', 'S', 'O']
    _morder = [1,3,2,0]
if cat == 'position':
    _morder = [2,3,0,1]

for var in ['totals_Y', 'totals_S']:
    print('{}_{}.png'.format(var, cat))
    f, ax = plt.subplots(figsize=(6,4), dpi=600)
    if cat == 'day':
        ax.xaxis.set_tick_params(labelrotation=70)
    if cat == 'daytime':
        ax.xaxis.set_ticklabels(['8 - 10', '11 - 13', '14 - 16', '17 - 21'])
    ax = swarmbox(x=cat, y=var, order=_order, m_order=_morder, palette=my_pal, data=totals, ax=ax)
    ax = set_font('Quicksand-Regular', ax=ax)
    ax.set_title(xlabels[cat], fontsize=10, loc='center', fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylabel(ylabels[var])
    sns.despine(ax=ax, bottom=True, trim=True)
    plt.tight_layout()
    plt.savefig(os.path.join(profile.out(), 'plots', '{}_{}.png'.format(cat, var)), dpi=600)
    plt.close()
"""

""" figues for timeseries
    f2, axo = plt.subplots(figsize=(8,8))
    f, axes = plt.subplots(4, figsize=(16,8), gridspec_kw = {'height_ratios':[3, 3, 3, 3]}, dpi=400)#{'height_ratios':[2, 0.75, 0.5, 3, 3, 3]}, dpi=600)
    # Trajectory
    axo = plot_fly(df, x='body_x', y='body_y', hx='head_x', hy='head_y', etho='etho', arena=meta['arena'], spots=meta['food_spots'], title=each.name, ax=axo)
    axo.get_xaxis().set_visible(False)
    axo.get_yaxis().set_visible(False)
    axo.spines['top'].set_visible(False)
    axo.spines['right'].set_visible(False)
    axo.spines['bottom'].set_visible(False)
    axo.spines['left'].set_visible(False)
    axo = set_font('Quicksand-Regular', ax=axo)


    # Ethogram
    ax = axes[0]
    for each_val in [-1,0,1,2,3,4,5,6]:#df['etho'].unique():
        ethocolor = {-1:'#7c7c7c', 0:'#404040', 1:'#7f00ac',2:'#ff6f6f', 3:'#4fff76', 4: '#ffc04c', 5: '#4c8bff', 6: '#ff0000'}
        ts = df.loc[df.query('etho == '+str(each_val)).index, 'time']
        ax.vlines(ts, 0, 1, colors=ethocolor[each_val], lw=0.5)#, zorder=each_val)
    ax.set_xlim([df.loc[df.index[0],'time'], df.loc[df.index[-1],'time']])
    ax.set_ylim([0, 1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_ylabel('Ethogram', labelpad=58, rotation= 0)
    ax = set_font('Quicksand-Regular', ax=ax)

    # Visits
    ax = axes[1]
    for each_val in [1,2]:#df['etho'].unique():
        subscolor = {1: '#ffc04c', 2: '#4c8bff'}
        ts = df.loc[df.query('visit == '+str(each_val)).index, 'time']
        ax.vlines(ts,0,1,colors=subscolor[each_val], alpha=0.5, lw=0.5)#, zorder=each_val)
    ax.set_xlim([df.loc[df.index[0],'time'], df.loc[df.index[-1],'time']])
    ax.set_ylim([0, 1])
    ax.set_ylabel('Visits', labelpad=58, rotation= 0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax = set_font('Quicksand-Regular', ax=ax)

    ### Encounters
    ax = axes[2]
    for each_val in [1,2]:#df['etho'].unique():
        subscolor = {1: '#ffae18', 2: '#206dfc'}
        ts2 = df.loc[df.query('encounter == '+str(each_val)).index, 'time']
        ax.vlines(ts2, 0, 0.75,colors=subscolor[each_val], alpha=1, lw=0.5)#, zorder=each_val)
    ax.set_xlim([df.loc[df.index[0],'time'], df.loc[df.index[-1],'time']])
    ax.set_ylim([0, 1])
    ax.set_ylabel('Encounters', labelpad=58, rotation= 0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax = set_font('Quicksand-Regular', ax=ax)

    ### Patch distance
    ax = axes[0]
    ax.hlines(5, df.loc[df.index[0],'time'], df.loc[df.index[-1],'time'], colors='#999999', linestyles='--', lw=1)
    ax.hlines(2.5, df.loc[df.index[0],'time'], df.loc[df.index[-1],'time'], colors='#999999', linestyles='--', lw=1)
    ax.plot(df['time'], df['min_dpatch'], color='#505050', ls='-')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xlim([df.loc[df.index[0],'time'], df.loc[df.index[-1],'time']])
    ax.set_ylim([0, 15])
    ax.set_ylabel('Min. dist.\nfrom patch\n[mm]')
    ax.get_xaxis().set_visible(False)
    ax = set_font('Quicksand-Regular', ax=ax)

    # Linear speed
    ax = axes[1]
    ax.hlines(2, df.loc[df.index[0],'time'], df.loc[df.index[-1],'time'], colors='#999999', linestyles='--', lw=1)
    ax.hlines(6, df.loc[df.index[0],'time'], df.loc[df.index[-1],'time'], colors='#999999', linestyles='--', lw=1)
    ax.plot(df['time'], df['sm_head_speed'], color='#600000', ls='-')
    ax.plot(df['time'], df['sm_body_speed'], color='#6f6464', ls='-')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_ylabel('Linear\nspeed\n[mm/s]')
    ax.set_xlim([df.loc[df.index[0],'time'], df.loc[df.index[-1],'time']])
    ax.set_yticks([0, 2, 4, 6, 8, 10, 15, 20])
    ax.set_ylim([0, 1.1*np.amax(np.array(df.loc[:,['sm_head_speed', 'sm_body_speed']]))])
    ax.get_xaxis().set_visible(False)
    ax = set_font('Quicksand-Regular', ax=ax)

    # Angular speed
    ax = axes[2]
    ax.hlines(125, df.loc[df.index[0],'time'], df.loc[df.index[-1],'time'], colors='#999999', linestyles='--', lw=1)
    ax.hlines(-125, df.loc[df.index[0],'time'], df.loc[df.index[-1],'time'], colors='#999999', linestyles='--', lw=1)
    ax.plot(df['time'], np.clip(df['angular_speed'], -500, 500), color='#004fff', ls='-')
    #ax.plot(df['sm_angular_speed'], color='#4a82ff', ls='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Angular\nspeed\n[º/s]')
    ax.spines['bottom'].set_visible(False)
    ax.set_xlim([df.loc[df.index[0],'time'], df.loc[df.index[-1],'time']])
    ax.set_yticks([-500, -250, 0, 250, 500])
    ax.set_ylim([-1.1*500, 1.1*500])
    ax.get_xaxis().set_visible(False)
    ax = set_font('Quicksand-Regular', ax=ax)


    ax = axes[3]
    ax.plot(df['time'], df['major'], color='#00ff6e', ls='-')
    ax.plot(df['time'], df['minor'], color='#d400ff', ls='-')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Major/minor\naxis\n[mm]')
    #ax.spines['bottom'].set_visible(False)
    ax.set_xlabel('Time elapsed [s]')
    ax.set_xlim([df.loc[df.index[0],'time'], df.loc[df.index[-1],'time']])
    ax.set_ylim([0., 4.])
    ax = set_font('Quicksand-Regular', ax=ax)

    plt.tight_layout()
    #plot_along(f2, axo)
    #plot_along(f, axes)
    f.savefig(os.path.join(profile.out(), 'plots', 'timeseries.png'), dpi=300)
    #f2.savefig(os.path.join(profile.out(), 'plots', 'trajectory.png'), dpi=600)
    plt.close()
"""
