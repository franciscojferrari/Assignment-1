import pandas as pd
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
sns.set(style = "darkgrid")
import matplotlib.pyplot as plt
# %matplotlib widget
# %matplotlib notebook
from sklearn import manifold
from adjustText import adjust_text
from sklearn.decomposition import PCA
from scipy.linalg import fractional_matrix_power
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy
from scipy.sparse.csgraph import floyd_warshall
from sklearn.neighbors import kneighbors_graph
import networkx as nx 


class assignment1:
    def __init__(self, df, target_name, K = 3):
        self.animals_df = df.drop(target_name, axis=1)
        self.target_column = df[target_name]
        self.K = K
        self.animals_np = self.animals_df.to_numpy().T
        self.rows, self.columns = self.animals_np.T.shape

    def pca_sklearn(self):
        self.skpca = PCA(n_components=self.K)
        self.animal_pca_sk = self.skpca.fit_transform(self.animals_df)
        self.animal_pca_sk_df = pd.DataFrame(data = self.animal_pca_sk)
        self.animal_pca_sk_df = pd.concat([self.animal_pca_sk_df, self.target_column], axis=1)
        return self.animal_pca_sk_df
        
    def pca_svd(self):
        #Calculate the column mean for the dataset
        self.columns_mean = np.average(self.animals_np.T, axis=0)
        self.col_means_row_vec = self.columns_mean.reshape((1, self.columns)) 
        # Center the dataset by substracting the mean
        self.animals_centered_np = (self.animals_np.T - self.col_means_row_vec).T

        #Singular Value Decomposition on the dataset
        self.u, self.s, self.vt = np.linalg.svd(self.animals_centered_np, full_matrices=False)
        # Get K Principal Components by selecting the K singular values and K left most singular vectors
        self.animals_k_pca = (np.diag(self.s[:self.K]) @ self.vt[:self.K])
        self.reconstructed_data = ((self.animals_k_pca.T @ self.u.T[:self.K]) + self.col_means_row_vec).T
        #Convert results from Numpy to Pandas
        self.animals_k_pca_df = pd.DataFrame(data = self.animals_k_pca.T)
        self.animals_k_pca_df = pd.concat([self.target_column, self.animals_k_pca_df], axis=1)

        # Calculate the explained variance for each singular value
        self.explained_variance_ = (self.s ** 2) / (self.rows - 1)
        total_var = self.explained_variance_.sum()
        #Calculate the explained variance ratio
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        return self.animals_k_pca_df

    def mds_from_similarity_matrix(self):
        #Compute th similarity matrix (GramMatrix)
        S = self.animals_np.T @ self.animals_np
        # Compute the PC scores from Gram matrix
        lamda, vl, vr  = linalg.eig(S, left=True, right=True)
        # Double check the eigenvalues are positive.
        assert np.all(lamda[:self.K] > 0)
    #     proj =fractional_matrix_power(np.diag(lamda[:K]) , 0.5) @ u.T[:K]
        proj = np.diag(np.sqrt(lamda[:self.K])) @ vl.T[:self.K]
        mds_proj = pd.DataFrame(data = proj.T.real).reset_index()
        mds_proj = pd.concat([self.target_column, mds_proj], axis=1)
        return mds_proj

    def __mds_from_distance_matrix(self, calculate_distance = False, D = None):
        if calculate_distance:
            D = pairwise_distances(D)
        centering_matrix = lambda n: np.identity(n) -(np.ones((n, 1)) @ np.ones((1, n)))/n
        
        N = D.shape[0]
        # Compute the Gram matrix from the distance_matrix
        gram_from_dist = -(centering_matrix(N) @ D @ centering_matrix(N))/2
        
        # Compute the eigen docomposition from Gram matrix
        lamda, vl, vr  = linalg.eig(gram_from_dist, left=True, right=True)
        # Double check the eigenvalues are positive.
        assert np.all(lamda[:self.K] > 0)
    #     proj =fractional_matrix_power(np.diag(lamda[:K]) , 0.5) @ u.T[:K]
        proj = np.diag(np.sqrt(lamda[:self.K])) @ vl.T[:self.K]

        mds_proj = pd.DataFrame(data = proj.T.real)
        mds_proj = pd.concat([ self.target_column, mds_proj], axis=1)
        return mds_proj, lamda, vl

    def mds_distance_matrix(self):
       self.animals_k_mds_df,lamda, vl = self.__mds_from_distance_matrix(True, self.animals_np.T)
       return self.animals_k_mds_df, lamda, vl

    def mds_distance_matrix_importance(self):
        self.calculate_information_gain()
        weighted_animals_he = self.animals_df.copy()
        for feature_name in self.animals_df:
            weighted_animals_he[feature_name] = \
                weighted_animals_he[feature_name] * \
                (self.information_gain_df [self.information_gain_df ['feature'] == feature_name]['information_gain'].values) * 10
            
        weighted_animals_he_np = weighted_animals_he.to_numpy()
        self.weighted_animals_k_mds_df, lamda, vl = self.__mds_from_distance_matrix(True, weighted_animals_he_np)
        return self.weighted_animals_k_mds_df, lamda, vl
        
    def iso_map(self, k_mean):
        self.k_means = k_mean
        self.nbrs_graph = kneighbors_graph(self.animals_np.T, self.k_means, mode='distance', include_self=False)
        dist_matrix = floyd_warshall(csgraph=self.nbrs_graph, directed=False)
        self.animals_k_isomap_df,lamda, vl  = self.__mds_from_distance_matrix(False, dist_matrix)
        return self.animals_k_isomap_df, lamda, vl

    def calculate_information_gain(self):
        '''
        Measures the reduction in entropy after the split  
        :param v: Pandas Series of the members
        :param split:
        :return:
        '''
        self.information_gain_df = pd.DataFrame()
        target = self.target_column.iloc[:,0]
        entropy_before = entropy( target.value_counts(normalize=True), base=2)
        
        for feature_name in self.animals_df:
            
            feature = self.animals_df[feature_name]
            
            grouped_distrib = target.groupby(feature) \
                                .value_counts(normalize=True) \
                                .reset_index(name='count') \
                                .pivot_table(index = feature_name, columns = self.target_column.columns[0], values='count').fillna(0) 

            entropy_after = entropy(grouped_distrib, axis=1, base=2)
            entropy_after *= feature.value_counts(sort=False, normalize=True)
            
            information_gain = entropy_before - entropy_after.sum()
            
            self.information_gain_df = self.information_gain_df.append(
                {"feature":feature_name, "information_gain":information_gain}
                , ignore_index=True)
                
            self.information_gain_df = self.information_gain_df.sort_values(by=['information_gain'], ascending=False)
        return self.information_gain_df
        
    def plot_graph(self, **kwargs):

        args = {'labels' : True, 'fsize' : (8,8), 'show_axis' : False, 'save':None, "mds_position":False, 'adjust_text':True}
        args.update(kwargs)

        G = nx.from_scipy_sparse_matrix(self.nbrs_graph)
        # Get Unique types
        color_labels = self.animals_k_isomap_df['type'].unique()
        # List of colors in the color palettes
        rgb_values = sns.color_palette("tab10", color_labels.size)
        # Map types to the colors
        color_map = dict(zip(color_labels, rgb_values))
        color_map_ = []
        for node in G:
            color_map_.append(self.animals_k_isomap_df['type'].map(color_map)[node])

        mapping = {x:(y+"_"+str(i)) for i,(x,y) in enumerate(zip(G, self.animals_k_isomap_df['animal']))}
        G = nx.relabel_nodes(G, mapping)

        if args['mds_position']:
           pos = dict(zip(G, self.animals_k_isomap_df[[0,1]].to_numpy()))
        else:
            pos = nx.spring_layout(G)
        
        d = dict(G.degree)

        fig = plt.figure(figsize=args['fsize'])
        ax = fig.add_subplot(111)
        # nx.draw(G, node_color=color_map_,  with_labels=True, node_size = 80, font_size=8)
        # if args['show_axis']:
        #     ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        nx.draw_networkx(G, pos=pos, node_color=color_map_, edge_color="grey", node_size=[v * 20 for v in d.values()], ax=ax, with_labels=False)
        if args['show_axis']:
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        # for node, (x, y) in pos.items():
        if args['labels']:
            texts =  [ ax.text(x, y, node, fontsize=7, ha='center', va='center') for node, (x, y) in pos.items()]
        if args['adjust_text']:
            adjust_text(texts)
        plt.draw()  # pyplot draw()
        plt.show()
        if args['save'] is not None:
            plt.savefig("{}network k ={}{}.png".format(args['save'], self.k_means, "_mds_position" if args['mds_position'] else ""), dpi=300)


def plot_scatter(data, title, **kwargs):

    args = {'ax':None, 'labels' : True, 'fsize' : (8,8), 'axis_text' : None, 'columns_to_plot' : ["0", "1"], 'threeD': False, 'save':None}
    args.update(kwargs)
    
    data.columns = data.columns.astype(str)
    x_name = args['columns_to_plot'][0]
    y_name = args['columns_to_plot'][1]
    if args['ax'] is  None:
        fig = plt.figure(figsize=args['fsize'])
        ax = fig.add_subplot(111, projection = "3d" if args['threeD'] else None)
    else:
        ax = args['ax']
    
    # Get Unique types
    color_labels = data['type'].unique()
    # List of colors in the color palettes
    rgb_values = sns.color_palette("tab10", color_labels.size)
    # Map types to the colors
    color_map = dict(zip(color_labels, rgb_values))
    
    for color_label in color_labels:
        
        data_by_type = data[data['type'] == color_label]
        if args['threeD']:
            z_name = args['columns_to_plot'][2]
            ax.scatter(
                data_by_type[x_name],
                data_by_type[y_name],
                data_by_type[z_name],
                c = data_by_type['type'].map(color_map),
                label=color_label)
            if args['labels']:
                texts = [ ax.text(
                            x = data_by_type.iloc[i,2],
                            y = data_by_type.iloc[i,3],
                            z = data_by_type.iloc[i,4],
                            s = text, fontsize=7,
                            ha='center', va='center')
                        for i, text in enumerate(data_by_type['animal'])
                        ]
                adjust_text(texts)
        else:
            ax.scatter(
                data_by_type[x_name],
                data_by_type[y_name],
                c = data_by_type['type'].map(color_map),
                label=color_label)
            if args['labels']:
                texts = [ ax.text(
                            x = data_by_type.iloc[i,2],
                            y = data_by_type.iloc[i,3],
                            s = text, fontsize=7,
                            ha='center', va='center')
                        for i, text in enumerate(data_by_type['animal'])
                        ]
                adjust_text(texts)
    if args['axis_text'] is not None:
        ax.set_xlabel(args['axis_text'] + x_name)
        ax.set_ylabel(args['axis_text'] + y_name)
    ax.set_title(title, fontsize=15)
    ax.legend(loc=0, prop={'size': 6})
    if args['save'] is not None:
        plt.savefig("{}{}.png".format(args['save'], title), dpi=300)
    return (ax)

