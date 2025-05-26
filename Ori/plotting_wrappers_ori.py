import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt,ceil
import matplotlib.animation as animation


def plot_boxes(df,title='',metric='',dots_colored_by=None):
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df, width=0.5, palette='Set3')
    if dots_colored_by==None:
        sns.stripplot(data=df, color='black', size=8, jitter=True, alpha=0.7)
    else:
        sns.stripplot(data=df, hue=dots_colored_by, size=8, jitter=True, alpha=0.7)
    # Set plot labels and title
    plt.ylabel(metric)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    # Show the plot
    plt.show()



def plot_scatter(x,y,title='',x_metric='',y_metric=''):
    plt.figure(figsize=(14, 6))
    sns.scatterplot(x=x,y=y)
    # Set plot labels and title
    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.title(title)

    # Show the plot
    plt.show()

def plot_jointplot(df,x_col,y_col,title='', equal_axis=True, axis_limits=None, groups_col=None, groups_order = None, continuous_color_scale=False, diag = True, toplot=True):
    continuous_color_palette = 'RdBu'
    #organize df
    required_cols = [x_col,y_col]
    if not groups_col==None:
        required_cols.append(groups_col)
    data_df = df[required_cols].dropna(axis=0).sort_values(by=x_col)

    #initialize the grid figure    
    #set params for discrete coloring
    '''
    if not groups_col==None and not continuous_color_scale:
        if groups_order==None:
            groups_order = data_df[groups_col].unique()
        g = sns.JointGrid(data=data_df, x=x_col, y=y_col, height=9, hue = groups_col, hue_order=groups_order)
    else:
        g = sns.JointGrid(data=data_df, x=x_col, y=y_col, height=9)
    '''
    g = sns.JointGrid(data=data_df, x=x_col, y=y_col, height=9)
    #plot the histogram plots
    g.plot_marginals(sns.histplot, kde=True,stat='probability')
    
    #plot the  scatter plot
    #use semi-opaque full circles if there's no coloring
    if groups_col==None:
        #g.plot_joint(sns.scatterplot, ec="b", fc="none", s=100, linewidth=1)
        g.plot_joint(sns.scatterplot, s=100, alpha=0.7)
    #add color bar if there's continuous coloring 
    elif continuous_color_scale:
        g.plot_joint(sns.scatterplot, data=data_df, s=100, alpha=0.7, hue=groups_col, palette=continuous_color_palette)
        norm = plt.Normalize(data_df[groups_col].min(), data_df[groups_col].max())
        sm = plt.cm.ScalarMappable(cmap=continuous_color_palette, norm=norm)
        sm.set_array([])
        g.ax_joint.get_legend().remove()
        g.ax_joint.figure.colorbar(sm)
        title = f'{title}\nColored by {groups_col}'
    #use semi-opaque full circles with hue if there's discrete coloring
    else:
        g.plot_joint(sns.scatterplot, data=data_df, s=100, alpha=0.7, hue = groups_col, hue_order=groups_order)

    # add diagonal line
    lim_down = min([data_df[x_col].min(),data_df[y_col].min()]) if axis_limits==None else axis_limits[0]
    lim_up = max([data_df[x_col].max(),data_df[y_col].max()]) if axis_limits==None else axis_limits[1]
    if diag:
        limits_list = [lim_down,lim_up]
        g.ax_joint.plot(limits_list, limits_list, 'k--', linewidth = 1)
    # add regression line
    slope, intercept, r2_adjusted, r2, y_pred = deming_regresion(data_df,x_col,y_col)

    x_edges = [data_df[x_col].min(),data_df[x_col].max()]
    x_edges_locs = [data_df[x_col].argmin(),data_df[x_col].argmax()]
    y_pred_edges = [y_pred[x_edges_locs[0]],y_pred[x_edges_locs[1]]]
    g.ax_joint.plot(x_edges, y_pred_edges, 'b-', linewidth = 1.5)
    
    # Set plot title. 
    intercept_sign = '+' if intercept>=0 else ''
    title_with_equation = f'{title}\n Y = {round(slope,2)}*X{intercept_sign}{round(intercept,2)} ;  R^2(adjusted) = {round(r2_adjusted,2)}'
    
    #x,y axis should have the same range if equal_axis is True
    ax_limits = (lim_down-0.001,lim_up+0.001)
    if equal_axis:    
        g.ax_joint.set(xlim=ax_limits,ylim=ax_limits)
    g.figure.suptitle(title_with_equation)
    if continuous_color_scale:
        g.figure.subplots_adjust(top=0.87,left=0.1, right=0.95)
    else:
        g.figure.subplots_adjust(top=0.9,left=0.1)
    if toplot:
        plt.show()
    else:
        return g


def plot_violin(df_val,df_dev=pd.DataFrame([]),df_val_name='Val',df_dev_name='Dev',title='',x_metric='X',y_metric='Y',inner='quart'):
    # inner = “box”, “quart”, “point”, “stick”, None
    #df_dev =  dev_auc_df.copy()
    #df_val = val_auc_df.copy()
    #title=''
    #x_metric='Set Size'
    #y_metric='AUC'

    set_col = 'set_name'
    df_val[set_col] = df_val_name
    if df_dev.values==[]:
        df0=df_val
    else:
        df_dev[set_col] = df_dev_name
        df0 = pd.concat([df_dev,df_val],axis=0,ignore_index = True)
    
    M = df0.shape[0]
    N = df0.shape[1]
    df1 = pd.DataFrame(np.zeros([M*(N-1),3]),columns = [x_metric,y_metric,set_col])
    for iter,col in enumerate(df0.columns[df0.columns!=set_col]):
        idx0 = M*iter
        idx1 = M*(iter+1)
        idx_range = np.arange(idx0,idx1)
        col_name_list = [col for idx in idx_range]
        
        df1.loc[df1.index.isin(idx_range),set_col]=df0.loc[:,set_col].values
        df1.loc[df1.index.isin(idx_range),x_metric]=col_name_list
        df1.loc[df1.index.isin(idx_range),y_metric]=df0.loc[:,col].values
    df1=df1.loc[df1.loc[:,y_metric]>0,:]
    plt.figure(figsize=(14, 6))
    if df_dev.values==[]:
        sns.violinplot(data=df1, x=x_metric, y=y_metric, inner=inner)
    else:
        sns.violinplot(data=df1, x=x_metric, y=y_metric, hue=set_col, split=True, gap=.1, inner=inner)
    # Set plot labels and title
    plt.title(title)
    # Show the plot
    plt.show()

def plot_differentiating_scatter(x,y,label,groups_order=None,title='',x_metric='',y_metric=''):
    if groups_order==None:
        groups_order=label.unique()
    
    plt.figure(figsize=(14, 6))
    
    # Set plot labels and title
    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.title(title)

    # Show the plot
    for grp in groups_order:
        sns.scatterplot(x=x[label==grp],y=y[label==grp])
    plt.legend(groups_order)
    plt.show()

def deming_regresion(df, x, y, delta = 1):
    '''Takes a pandas DataFrame, name of the 
    columns as strings and the value of delta, 
    and returns the slope and intercept following deming regression formula'''
    # covariance matrix and average calculations
    cov = df[[x,y]].cov()
    mean_x = df[x].mean()
    mean_y = df[y].mean()
    s_xx = cov[x][x]
    s_yy = cov[y][y]
    s_xy = cov[x][y]
    #linear equation coefficients
    slope = (s_yy  - delta * s_xx + np.sqrt((s_yy - delta * s_xx) ** 2 + 4 * delta * s_xy ** 2)) / (2 * s_xy)
    intercept = mean_y - slope  * mean_x
    y_pred = [slope*xi+intercept for xi in df[x].values]
    
    #fitting error
    e = np.array([line_point_distance(xi,yi,slope,intercept)**2 for (xi,yi) in zip(df[x].values,df[y].values)])
    e_vertical = np.array([(yi-y_pred_i)**2 for (yi,y_pred_i) in zip(df[y].values,y_pred)])
    #calculate r squared according to the adjustments needed for a Deming regression process
    r2_adjusted = 1-sum(e)/(df.shape[0]*(s_xx+s_yy-abs(s_xy)))
    r2 = 1-sum(e_vertical)/(df.shape[0]*(s_yy))
    print("R^2 (adjusted):",round(r2_adjusted,3),",R^2:",round(r2,3),",R^2 diff:",round(r2_adjusted-r2,4))

    return slope, intercept, r2_adjusted,r2, y_pred

def line_point_distance(x0,y0, slope,intercept):
    e = abs(slope*x0-y0+intercept)/sqrt(slope**2+1)
    return e

def pairplot_wrapper(df,label,groups_order=None,title=''):
    if groups_order==None:
        groups_order=label.unique()
    df_plot=pd.concat([df,label],axis=1)
    
    pp=sns.pairplot(df_plot,hue=label.name,hue_order=groups_order)
    pp.figure.suptitle(title)
    pp.figure.subplots_adjust(top=0.9)
    plt.show()

def pad_image(image_arr,sz_x,sz_y,value=[255,255,255]):
    #rounding size to 16*16 blocks
    macro_block_size=16
    sz_x = ceil(sz_x/macro_block_size)*macro_block_size
    sz_y = ceil(sz_y/macro_block_size)*macro_block_size
    
    sz = image_arr.shape
    pad_sz_x_right = int((sz_x-sz[1])/2)
    pad_sz_x_left = sz_x-sz[1]-pad_sz_x_right
    pad_sz_y_top = int((sz_y-sz[0])/2)
    pad_sz_y_bottom = sz_y-sz[0]-pad_sz_y_top

    pad_right = np.array([[value for num1 in range(pad_sz_x_right)] for num2 in range(pad_sz_y_top+pad_sz_y_bottom+sz[0])])
    pad_left = np.array([[value for num1 in range(pad_sz_x_left)] for num2 in range(pad_sz_y_top+pad_sz_y_bottom+sz[0])])
    pad_top = np.array([[value for num1 in range(sz[1])] for num2 in range(pad_sz_y_top)])
    pad_bottom = np.array([[value for num1 in range(sz[1])] for num2 in range(pad_sz_y_bottom)])

    image_arr_out = np.concatenate([pad_left,np.concatenate([pad_top,image_arr,pad_bottom],axis=0),pad_right],axis=1)

    return image_arr_out
    
#def plot_stacked_bars(df, x_group_column, stacked_group_column) TODO