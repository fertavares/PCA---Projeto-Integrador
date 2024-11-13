import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_diabetes

class PCAAcessibilidade:
    
    def __init__(self, caminho):
        try:
            # self.data = pd.read_csv(caminho, encoding="utf-8", low_memory=False)
            self.data = load_diabetes()

            self.df = pd.DataFrame(self.data.data, columns=self.data.feature_names)
            print(self.df) 
        except Exception as e:
            self.data = None
            print(f"[Erro] Não foi possível carregar o arquivo: {e}")
                
    def limpar_data(self):
        if self.df is not None:
            
            required_columns = ["s1", "s2", "s3", "s4", "s5", "s6"] 
            if all(col in self.df.columns for col in required_columns):
                self.acessibilidade = self.df[required_columns].dropna().drop_duplicates().reset_index(drop=True)
                print(self.acessibilidade.head())

            else:
                print("[Erro] Colunas de acessibilidade não encontradas.")
        else:
            print("[Erro] Dados não carregados corretamente.")
        
    def calcular_pca(self):
        if hasattr(self, 'acessibilidade') and not self.acessibilidade.empty:

            self.acessibilidade = self.acessibilidade.apply(pd.to_numeric, errors='coerce').dropna()

            self.pca_pipeline = make_pipeline(StandardScaler(), PCA(n_components=2))

            pca_results = self.pca_pipeline.fit_transform(self.acessibilidade)
            self.pca_df = pd.DataFrame(data=pca_results, columns=['PC1', 'PC2'])
            
            print(self.pca_df.info())

            self.pca_df['Categoria Acessibilidade'] = self.acessibilidade.apply(
                lambda row: ', '.join([f"{col}: {val}" for col, val in row.items()]), axis=1
            )
        else:
            print("[Erro] Não há dados válidos para realizar a PCA.")
    
    def plot_pca(self):
        if hasattr(self, 'pca_df'):
            plt.figure(figsize=(10, 8))
            
            scatter = plt.scatter(
                self.pca_df['PC1'], self.pca_df['PC2'],
                c=self.pca_df['Categoria Acessibilidade'].astype('category').cat.codes, 
                cmap='tab10', marker="o", edgecolor="k"
            )

            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title('PCA das Colunas de Acessibilidade')

            labels = self.pca_df['Categoria Acessibilidade'].unique()
            handles, _ = scatter.legend_elements()
            plt.legend(handles, labels, title="Categoria Acessibilidade", bbox_to_anchor=(1.05, 1), loc='upper left')

            pca = self.pca_pipeline.named_steps['pca']
            coeff = np.transpose(pca.components_)
            for i in range(len(coeff)):
                plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='red', alpha=0.5, head_width=0.05, head_length=0.1)
                plt.text(coeff[i, 0] * 1.5, coeff[i, 1] * 1.5, f"Componente {i+1}", color='red', ha='center', va='center')

            plt.show()
        else:
            print("[Erro] PCA não calculado ou dados ausentes.")


if __name__ == "__main__":
    caminho = r"..\trabalhopca\DFmicrodados2021.csv"
    pca = PCAAcessibilidade(caminho)
    pca.limpar_data()
    pca.calcular_pca()
    pca.plot_pca()
